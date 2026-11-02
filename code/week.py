"""Weekly episodes: a fixed courier roster works consecutive days; deficit queues, earnings and
online time carry over, vehicles (consumption class, pack type) are persistent."""
import hashlib
import pickle
from pathlib import Path

import numpy as np

from sim import Sim, site_cabinets

ROOT = Path(__file__).resolve().parents[1]
_INST = {}

SUM_KEYS = ["orders", "served", "ontime", "pickup_delay_sum", "swaps", "stockouts", "res_fail", "ext_lost", "ext_served", "strand"]


def inst(dist):
    if dist not in _INST:
        _INST[dist] = pickle.load(open(ROOT / "data" / "instances" / f"{dist}.pkl", "rb"))
    return _INST[dist]


def weeks(dist, n_cal=2, n_test=4, length=7):
    """Consecutive blocks of `length` kept days; first n_cal blocks calibrate, the next n_test test."""
    days = sorted(inst(dist)["days"])
    blocks = [days[i:i + length] for i in range(0, len(days) - length + 1, length)]
    # spread test weeks over the period
    cal = blocks[:n_cal]
    rest = blocks[n_cal:]
    idx = np.linspace(0, len(rest) - 1, n_test).round().astype(int)
    return cal, [rest[i] for i in idx]


def run_week(dist, days, cfg, pol, keep_trace=False, rho_eval=None):
    I = inst(dist)
    n = max(3, int(np.ceil(np.mean([len(I["days"][d]["t"]) for d in days]) / cfg.fleet_per_orders)))
    rng = np.random.default_rng([cfg.seed, 99, int(days[0])])
    persona = dict(n=n, ptype=(rng.random(n) < cfg.share_b).astype(int), heavy=rng.random(n) < cfg.share_heavy,
                   earn=np.zeros(n), Q=np.zeros(n), online=np.zeros(n), credit=np.zeros(n))
    persona["gamer"] = np.random.default_rng([cfg.seed, 55, int(days[0])]).random(n) < cfg.gamer_share
    # one cabinet network for the whole week, sited on the held-out calibration week's requests
    x0, x1, y0, y1 = I["bbox"]
    n_cab = cfg.n_cab or max(2, int(np.ceil(n / cfg.couriers_per_cab)), int(np.ceil(0.6 * (x1 - x0) * (y1 - y0) / cfg.km2_per_cab)))
    cal = [d for wk in weeks(dist)[0] for d in wk]
    X = np.vstack([np.c_[I["days"][d]["x"], I["days"][d]["y"]] for d in cal])
    # network drawn per seed and week (k-means start and type-B stocking), sited on calibration-period demand
    persona["cab"], persona["has_b"] = site_cabinets(X, n_cab, I["bbox"], cfg, np.random.default_rng([cfg.seed, 77, int(days[0])]))
    tot = {key: 0.0 for key in SUM_KEYS}
    waits, sww, late, kmo, ms, trace = [], [], [], [], [], []
    overtime, unfinished, shift_h, pay_all = 0.0, 0, 0.0, 0.0
    km_i = np.zeros(n)
    path = hashlib.sha1()  # fingerprint of assignments, earnings, and end-of-day physical state
    for d in days:
        s = Sim(I, d, cfg, pol, persona=persona)
        m, _ = s.run()
        for key in SUM_KEYS:
            tot[key] += m[key]
        waits.append(m["mean_wait"] * m["served"]); late.append(m["late_rate"] * m["served"])
        sww += s.swap_wait; kmo.append(s.km_total); ms += s.dec_ms
        overtime += float(s.overtime_min.sum()); unfinished += s.unfinished
        shift_h += float((s.s_off - s.s_on).sum()) / 60.0; pay_all += float(s.o_pay.sum())
        if keep_trace:
            trace += [(d,) + x for x in s.trace]
        assert m["deficit_ok"]
        km_i += s.km_i
        for arr in (s.o_who, s.o_state, s.earn, s.E, s.xy, np.concatenate(s.pk_E), np.concatenate(s.pk_res)):
            path.update(np.ascontiguousarray(arr).tobytes())
        persona.update(earn=s.earn, Q=s.Q, online=s.online_min, credit=s.credit_min, E_end=s.E.copy(), xy_end=s.xy.copy())
    hrs = np.maximum(persona["online"] / 60.0, 1e-9)
    rate = persona["earn"] / hrs
    r = np.sort(rate)
    k = len(r)
    rho = pol.rho if rho_eval is None else rho_eval
    dis = persona["heavy"] | (persona["ptype"] == 1)
    out = dict(dist=dist, week=days[0], policy=pol.name, seed=cfg.seed, fleet=n,
               service_rate=tot["served"] / tot["orders"], ontime_rate=tot["ontime"] / tot["orders"],
               pickup_delay=tot["pickup_delay_sum"] / max(tot["served"], 1), cabinets=int(n_cab), mean_wait=sum(waits) / max(tot["served"], 1),
               late_rate=sum(late) / max(tot["served"], 1), km_per_order=sum(kmo) / max(tot["served"], 1),
               swaps_per_courier_day=tot["swaps"] / (n * len(days)), swap_wait=float(np.mean(sww)) if sww else 0.0,
               swap_wait_p90=float(np.percentile(sww, 90)) if sww else 0.0,
               stockouts=tot["stockouts"], res_fail=tot["res_fail"], ext_lost=tot["ext_lost"],
               ext_lost_share=tot["ext_lost"] / max(tot["ext_lost"] + tot["ext_served"], 1), strand=tot["strand"],
               earn_total=float(persona["earn"].sum()),
               gini=float((2 * np.arange(1, k + 1) - k - 1).dot(r) / (k * r.sum())),
               jain=float(r.sum() ** 2 / (k * (r ** 2).sum())), rate_mean=float(r.mean()),
               rate_min=float(r[0]), rate_p10=float(np.percentile(r, 10)), rate_p90=float(np.percentile(r, 90)),
               deficit_end_mean=float(np.maximum(rho * persona["online"] / 60.0 - persona["earn"], 0).mean()) if rho > 0 else np.nan,
               floor_viol=float((rate < rho - 1e-6).mean()) if rho > 0 else np.nan,
               shortfall=float(np.maximum(rho - rate, 0).mean()) if rho > 0 else np.nan,
               rate_adv=float(rate[~dis].mean()) if (~dis).any() else np.nan,
               rate_dis=float(rate[dis].mean()) if dis.any() else np.nan,
               gap_dis=float(rate[~dis].mean() - rate[dis].mean()) if dis.any() and (~dis).any() else np.nan,
               overtime_min_per_courier_day=overtime / (n * len(days)), unfinished=unfinished,
               rate_ub=pay_all / shift_h, floor_infeasible_cert=bool(rho > pay_all / shift_h) if rho > 0 else False,
               topup_per_courier_week=float((np.maximum(rho - rate, 0) * hrs).mean()) if rho > 0 else np.nan,
               topup_credit_per_courier_week=float(np.maximum(rho * persona["credit"] / 60.0 - persona["earn"], 0).mean()) if rho > 0 else np.nan,
               credit_share=float(persona["credit"].sum() / max(persona["online"].sum(), 1e-9)),
               q_end_mean=float(persona["Q"].mean()), q_end_max=float(persona["Q"].max()),
               path_hash=path.hexdigest(), dec_ms_mean=float(np.mean(ms)), dec_ms_p99=float(np.percentile(ms, 99)), dec_ms_max=float(np.max(ms)))
    extra = dict(km=km_i, rate=rate, hours=hrs, credit=persona["credit"] / 60.0, earn=persona["earn"], gamer=persona["gamer"], heavy=persona["heavy"], ptype=persona["ptype"], trace=trace, Q=persona["Q"])
    return out, extra
