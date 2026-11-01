"""Small-instance hindsight oracle (MILP, HiGHS via scipy.optimize.milp).

Two-hour windows, a handful of couriers with full packs (no swap is needed, so the oracle and
the online policies face the same energy-feasible problem). The oracle knows every request in
advance and may pre-position couriers, so it upper-bounds what any online policy can reach.
Objectives: (eff) total utility sum(pay) - c_km*km; (fair) max-min earnings rate (+eps*pay).
"""
import json
import sys
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

sys.path.insert(0, str(Path(__file__).parent))
from sim import Cfg, Policy, Sim  # noqa: E402
from week import inst  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_SEL = json.load(open(ROOT / "results" / "calibration.json")) if (ROOT / "results" / "calibration.json").exists() else {}
VSTAR, LAM, LAMR = _SEL.get("V", 0.3), _SEL.get("lam", 0.3), _SEL.get("lamr", _SEL.get("lam", 0.3))


def small_instance(dist, day, w0, n_cour, cfg, n_orders=24):
    I = inst(dist)
    d = I["days"][day]
    w1 = w0 + 120
    sel = np.where((d["t"] >= w0 + cfg.lead) & (d["t"] < w1 - 30 + cfg.lead))[0]
    rng = np.random.default_rng([cfg.seed, int(day), w0])
    sel = np.sort(rng.choice(sel, min(n_orders, len(sel)), replace=False))  # keep the MILP exactly solvable
    shifts = np.tile([[1, w0 + 15, w1 - 15]], (n_cour, 1)).astype(np.int32)
    return {"hub": I["hub"], "bbox": I["bbox"],
            "days": {day: {"t": d["t"][sel], "x": d["x"][sel], "y": d["y"][sel], "shifts": shifts}}}, w0, w1


def solve(sim, objective, time_limit=300, min_rate_floor=None, util_floor=None):
    """objective eff: max utility; fair: max min rate. Optional floors make lexicographic and frontier solves."""
    c = sim.cfg
    J, N = sim.n_o, sim.n
    rel, dl, pay, oxy = sim.o_rel, sim.o_dl, sim.o_pay, sim.o_xy
    ttm = lambda km: np.ceil(km / c.speed)  # planning travel time (no noise)
    Doo = c.detour * np.hypot(oxy[:, None, 0] - oxy[None, :, 0], oxy[:, None, 1] - oxy[None, :, 1])
    Dso = c.detour * np.hypot(sim.xy[:, None, 0] - oxy[None, :, 0], sim.xy[:, None, 1] - oxy[None, :, 1])
    budget = sim.E - c.reserve * sim.capw
    arcs = []  # (courier, u, v, km); u=-1 start, v=J sink
    for i in range(N):
        early = np.maximum(rel, sim.s_on[i] + ttm(Dso[i]))
        for v in range(J):
            if sim.s_on[i] + ttm(Dso[i, v]) <= dl[v]:
                arcs.append((i, -1, v, Dso[i, v]))
        arcs.append((i, -1, J, 0.0))
        for u in range(J):
            arcs.append((i, u, J, 0.0))
            for v in range(J):
                if u != v and early[u] + c.t_service + ttm(Doo[u, v]) <= dl[v]:
                    arcs.append((i, u, v, Doo[u, v]))
    A = len(arcs)
    nx = A
    iT = nx            # T_j service-start times
    iz = nx + J        # z for max-min
    nv = iz + 1
    cost = np.zeros(nv)
    for a, (i, u, v, km) in enumerate(arcs):
        gain = pay[v] if v < J else 0.0
        if objective == "eff":
            cost[a] = -(gain - c.c_km * km)
        else:
            cost[a] = 0.0  # pure max-min: the dual bound then bounds the minimum rate itself
    if objective == "fair":
        cost[iz] = -1.0
    rows, lb, ub = [], [], []
    M = lil_matrix((N + N * J + J + len(arcs) + N + N + 1, nv))
    r = 0
    by_i_u = {}
    for a, (i, u, v, km) in enumerate(arcs):
        by_i_u.setdefault((i, u), []).append(a)
    by_i_v = {}
    for a, (i, u, v, km) in enumerate(arcs):
        by_i_v.setdefault((i, v), []).append(a)
    for i in range(N):  # leave start exactly once (possibly straight to sink)
        for a in by_i_u[(i, -1)]:
            M[r, a] = 1
        lb.append(1); ub.append(1); r += 1
    for i in range(N):  # flow conservation at orders
        for j in range(J):
            for a in by_i_v.get((i, j), []):
                M[r, a] += 1
            for a in by_i_u.get((i, j), []):
                M[r, a] -= 1
            lb.append(0); ub.append(0); r += 1
    for j in range(J):  # served at most once
        for i in range(N):
            for a in by_i_v.get((i, j), []):
                M[r, a] = 1
        lb.append(0); ub.append(1); r += 1
    big = 2000.0
    for a, (i, u, v, km) in enumerate(arcs):  # time propagation
        if v == J:
            continue
        if u == -1:  # T_v >= s_i + tt - big(1-x)
            M[r, iT + v] = 1; M[r, a] = -big
            lb.append(sim.s_on[i] + ttm(km) - big); ub.append(np.inf)
        else:
            M[r, iT + v] = 1; M[r, iT + u] = -1; M[r, a] = -big
            lb.append(c.t_service + ttm(km) - big); ub.append(np.inf)
        r += 1
    for i in range(N):  # energy budget per courier
        for a in [a for a, (ii, u, v, km) in enumerate(arcs) if ii == i]:
            M[r, a] = arcs[a][3] * sim.whkm[i]
        lb.append(-np.inf); ub.append(budget[i]); r += 1
    if util_floor is not None:  # sum of arc utilities >= util_floor
        for a, (i, u, v, km) in enumerate(arcs):
            M[r, a] = (pay[v] if v < J else 0.0) - c.c_km * km
        lb.append(util_floor); ub.append(np.inf); r += 1
    if objective == "fair" or min_rate_floor is not None:
        H = (sim.s_off - sim.s_on) / 60.0
        for i in range(N):  # earnings_i - z*H_i >= 0
            for a in [a for a, (ii, u, v, km) in enumerate(arcs) if ii == i and v < J]:
                M[r, a] = pay[arcs[a][2]]
            M[r, iz] = -H[i]
            lb.append(0); ub.append(np.inf); r += 1
    M = M[:r].tocsr()
    lo = np.r_[np.zeros(A), rel.astype(float), 0.0 if min_rate_floor is None else float(min_rate_floor)]
    # service must finish within every courier's shift: all couriers share one window here
    hi = np.r_[np.ones(A), np.minimum(dl, sim.s_off.max() - c.t_service).astype(float), np.inf]
    integ = np.r_[np.ones(A), np.zeros(J + 1)]
    res = milp(cost, constraints=LinearConstraint(M, lb, ub), integrality=integ, bounds=Bounds(lo, hi),
               options=dict(time_limit=time_limit, mip_rel_gap=1e-3))
    if res.x is None:
        return None
    x = res.x[:A] > 0.5
    earn = np.zeros(N)
    util = 0.0
    served = 0
    for a in np.where(x)[0]:
        i, u, v, km = arcs[a]
        if v < J:
            earn[i] += pay[v]; served += 1
        util += (pay[v] if v < J else 0.0) - c.c_km * km
    H = (sim.s_off - sim.s_on) / 60.0
    bound = -float(getattr(res, "mip_dual_bound", np.nan))  # valid upper bound on the maximization objective
    return dict(util=util, served=served, min_rate=float((earn / H).min()), gap=getattr(res, "mip_gap", np.nan),
                bound=bound, status=res.status, arcs=A)


def online(sim_inst, day, cfg, pol, n):
    """Online policy on the oracle instance, scored like the oracle: only pickups whose service ends within the
    shift count, and hours are the shift length (the MILP's feasible set contains every such online schedule)."""
    s = Sim(sim_inst, day, cfg, pol, fleet=n)
    s.run(t0=int(s.s_on.min()), t1=int(s.s_off.max()) + 60)
    done = (s.o_state == 2) & (s.o_arrive + cfg.t_service <= s.s_off.max())
    who = np.full(s.n_o, -1)
    for j in np.where(done)[0]:
        who[j] = s.o_who[j]
    earn = np.array([s.o_pay[(who == i)].sum() for i in range(s.n)])
    util = float(s.o_pay[done].sum() - cfg.c_km * s.km_total)
    H = (s.s_off - s.s_on) / 60.0
    return dict(util=util, served=int(done.sum()), min_rate=float((earn / H).min()))


def one(a):
    dist, day, w0, n, seed, rho = a
    cfg = replace(Cfg(seed=seed, noise=0.0), soc0=(0.95, 0.95), load_cap=99, ext_rate=0.0)
    I, w0, w1 = small_instance(dist, day, w0, n, cfg)
    base = Sim(I, day, cfg, Policy("x"), fleet=n)
    out = dict(dist=dist, day=day, w0=w0, couriers=n, orders=base.n_o, seed=seed)
    for obj in ("eff", "fair"):
        r = solve(base, obj)
        out.update({f"orc_{obj}_{k}": v for k, v in (r or {}).items()})
    for name, pol in [("EFF", Policy("EFF", "eff", "thresh")), ("FFJ", Policy("FFJ", "ff", "joint", lam=LAM)),
                      ("FF-JR", Policy("FF-JR", "ff_rate", "joint", lam=LAMR)),
                      ("J-EFF", Policy("J-EFF", "dpp", "joint", V=VSTAR, rho=0.0)),
                      ("OURS", Policy("OURS", "dpp", "joint", V=VSTAR, rho=rho))]:
        r = online(I, day, cfg, pol, n)
        out.update({f"{name}_{k}": v for k, v in r.items()})
    return out


if __name__ == "__main__":
    import json
    sel = json.load(open(ROOT / "results" / "calibration.json"))
    jobs = []
    for dist in ["SH-43", "YT-125", "JL-core", "SH-5"]:
        days = sorted(inst(dist)["days"])
        for k, day in enumerate(days[20:80:15]):
            jobs.append((dist, day, 10 * 60, 3, k, sel["rho"][dist]))
    with Pool(8) as p:
        out = p.map(one, jobs, chunksize=1)
    df = pd.DataFrame(out)
    df.to_csv(ROOT / "results" / "E11_oracle.csv", index=False)
    print(df.filter(regex="orders|util|min_rate|gap").round(2).to_string())
