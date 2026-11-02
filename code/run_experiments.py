"""All paper experiments. Usage: python3 -I code/run_experiments.py [cal|E1|E2|...|all]

Calibration (held-out first week of each district) fixes rho, V* and lambda* before any test run.
"""
import json
import sys
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from sim import Cfg, Policy  # noqa: E402
from week import run_week, weeks  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
DISTS = ["SH-5", "SH-43", "YT-111", "YT-125", "JL-core"]
SEEDS = [0, 1, 2]
KAPPA = 0.9  # earnings floor = KAPPA x mean earnings rate of J-EFF on the calibration week
V_GRID, LAM_GRID = [0.01, 0.03, 0.1, 0.3, 1, 3, 10], [0.3, 1, 3, 10, 30, 100]


def policies(rho, V, lam, lamr=None):
    lamr = lam if lamr is None else lamr
    return {
        "NEAR": Policy("NEAR", "near", "thresh"),
        "EFF": Policy("EFF", "eff", "thresh"),
        "AGN": Policy("AGN", "eff", "thresh", energy_filter=False),
        "SEQ": Policy("SEQ", "eff", "seq"),
        "FF": Policy("FF", "ff", "thresh", lam=lam),
        "FFJ": Policy("FFJ", "ff", "joint", lam=lam),
        "FF-JR": Policy("FF-JR", "ff_rate", "joint", lam=lamr),
        "EFF-R": Policy("EFF-R", "dpp", "thresh", V=V, rho=0.0),
        "SEQ-R": Policy("SEQ-R", "dpp", "seq", V=V, rho=0.0),
        "J-EFF": Policy("J-EFF", "dpp", "joint", V=V, rho=0.0),
        "DPP-THR": Policy("DPP-THR", "dpp", "thresh", V=V, rho=rho),
        "OURS-noRes": Policy("OURS-noRes", "dpp", "joint_nores", V=V, rho=rho),
        "OURS": Policy("OURS", "dpp", "joint", V=V, rho=rho),
        "J-EFF-0": Policy("J-EFF-0", "dpp", "joint", V=V, rho=0.0, ctg=0.0),
        "EFF-R0": Policy("EFF-R0", "dpp", "thresh", V=V, rho=0.0, ctg=0.0),
        "SEQ-R-noLock": Policy("SEQ-R-noLock", "dpp", "seq_nores", V=V, rho=0.0),
        "J-EFF-noLock": Policy("J-EFF-noLock", "dpp", "joint_nores", V=V, rho=0.0),
        "OURS-noQswap": Policy("OURS-noQswap", "dpp", "joint", V=V, rho=rho, qswap=False),
        "IFD-Q": Policy("IFD-Q", "dpp", "joint", V=V, rho=rho, signal="queue"),
        "FF-JR-0": Policy("FF-JR-0", "ff_rate", "joint", lam=lamr, ctg=0.0),
        "FF-JR-10": Policy("FF-JR-10", "ff_rate", "joint", lam=10.0),  # interior weight, compared on the same seeds and weeks
        "OURS-0": Policy("OURS-0", "dpp", "joint", V=V, rho=rho, ctg=0.0),
        "EFF-J": Policy("EFF-J", "eff", "joint"),  # joint reservation columns without rate normalization (RR-31)
        "OURS-C": Policy("OURS-C", "dpp", "joint", V=V, rho=rho, credit=True),  # IFD on credited time
        "FF-JR-C": Policy("FF-JR-C", "ff_rate", "joint", lam=lamr, credit=True),
    }


def job(a):
    exp, dist, days, cfg, pol, rho_eval, tag = a
    o, x = run_week(dist, days, cfg, pol, keep_trace=(exp in ("E9", "E14")), rho_eval=rho_eval)
    o.update(exp=exp, tag=tag)
    rates = pd.DataFrame(dict(courier=np.arange(len(x["rate"])), rate=x["rate"], hours=x["hours"], credit=x["credit"], earn=x["earn"], gamer=x["gamer"], heavy=x["heavy"], ptype=x["ptype"], km=x["km"]))
    for k in ("exp", "dist", "week", "seed", "policy", "tag"):
        rates[k] = o[k]
    tr = pd.DataFrame(x["trace"], columns=["day", "t", "q_mean", "q_max", "online"]) if x["trace"] else None
    if tr is not None:
        for k in ("dist", "week", "seed", "policy", "tag"):
            tr[k] = o[k]
    return o, rates, tr


def pool_run(jobs, name):
    with Pool(16) as p:
        out = p.map(job, jobs, chunksize=1)
    df = pd.DataFrame([o for o, _, _ in out])
    df.to_csv(RES / f"{name}.csv", index=False)
    pd.concat([r for _, r, _ in out]).to_csv(RES / f"{name}_rates.csv", index=False)
    trs = [t for _, _, t in out if t is not None]
    if trs:
        pd.concat(trs).to_csv(RES / f"{name}_trace.csv", index=False)
    print(name, len(df), "episodes")
    return df


def calibrate():
    base = Cfg()
    jobs = []
    for d in DISTS:
        for cal in weeks(d)[0]:
            for s in SEEDS:
                jobs.append(("cal", d, cal, replace(base, seed=s), Policy("J-EFF", "dpp", "joint", V=1, rho=0.0), None, "jeff"))
    df = pool_run(jobs, "cal_rho")
    rho = {d: KAPPA * df[df.dist == d].rate_mean.mean() for d in DISTS}
    jobs = []
    for d in DISTS:
      for cal in weeks(d)[0]:
        for s in SEEDS:
            cfg = replace(base, seed=s)
            jobs.append(("cal", d, cal, cfg, Policy("J-EFF", "dpp", "joint", V=1, rho=0.0), rho[d], "jeff"))
            for V in V_GRID:
                jobs.append(("cal", d, cal, cfg, Policy("OURS", "dpp", "joint", V=V, rho=rho[d]), rho[d], f"V={V}"))
            for lam in LAM_GRID:
                jobs.append(("cal", d, cal, cfg, Policy("FFJ", "ff", "joint", lam=lam), rho[d], f"lam={lam}"))
                jobs.append(("cal", d, cal, cfg, Policy("FF-JR", "ff_rate", "joint", lam=lam), rho[d], f"lamr={lam}"))
    df = pool_run(jobs, "cal_select")
    g = df.groupby("tag")[["service_rate", "shortfall", "mean_wait", "gini"]].mean().round({"shortfall": 4, "gini": 4})
    ref = g.loc["jeff", "service_rate"]

    def pick(prefix):  # min shortfall s.t. service within 0.5 pt of J-EFF; ties -> lower Gini, then lower mean wait
        c = g[g.index.str.startswith(prefix) & (g.service_rate >= ref - 0.005)]
        c = c if len(c) else g[g.index.str.startswith(prefix)]
        best = c.sort_values(["shortfall", "gini", "mean_wait"]).index[0]  # ties: fairer (lower Gini), then faster
        return float(best.split("=")[1])

    sel = dict(rho=rho, V=pick("V="), lam=pick("lam="), lamr=pick("lamr="), kappa=KAPPA, table=g.round(4).to_dict())
    json.dump(sel, open(RES / "calibration.json", "w"), indent=1)
    print(json.dumps({k: sel[k] for k in ("rho", "V", "lam", "lamr")}, indent=1))
    return sel


def load_cal():
    return json.load(open(RES / "calibration.json"))


def grid(exp, pols, cfgs, seeds=SEEDS, n_weeks=4, rho_override=None):
    """cfgs: list of (tag, Cfg-overrides dict); pols: list of policy names."""
    sel = load_cal()
    jobs = []
    for d in DISTS:
        rho = sel["rho"][d] if rho_override is None else rho_override * sel["rho"][d] / KAPPA
        P = policies(rho, sel["V"], sel["lam"], sel.get("lamr"))
        for w in weeks(d)[1][:n_weeks]:
            for s in seeds:
                for tag, ov in cfgs:
                    cfg = replace(Cfg(), seed=s, **ov)
                    for name in pols:
                        pol = P[name] if not callable(name) else name(rho)
                        jobs.append((exp, d, w, cfg, pol, rho, tag))
    return pool_run(jobs, exp)


ALL = ["NEAR", "EFF", "AGN", "SEQ", "FF", "FFJ", "EFF-R", "SEQ-R", "J-EFF", "FF-JR", "DPP-THR", "OURS-noRes", "OURS",
       "J-EFF-0", "FF-JR-0", "OURS-0", "EFF-R0", "SEQ-R-noLock", "J-EFF-noLock", "OURS-noQswap", "IFD-Q"]
NOCTG = ["J-EFF-0", "FF-JR-0", "OURS-0"]
CORE = ["EFF", "SEQ", "FFJ", "J-EFF", "FF-JR", "OURS", "IFD-Q"]


def run(which):
    RES.mkdir(exist_ok=True)
    if which in ("cal", "all"):
        calibrate()
    sel = load_cal()
    if which in ("E1", "all"):  # main comparison
        grid("E1", ALL, [("default", {})])
    if which in ("E2", "all"):  # trade-off curves
        mk = []
        for V in [0.03, 0.1, 0.3, 1, 3, 10, 30, 100]:
            mk.append(lambda rho, V=V: Policy(f"OURS V={V}", "dpp", "joint", V=V, rho=rho))
        for lam in [0.1, 0.3, 1, 3, 10]:
            mk.append(lambda rho, lam=lam: Policy(f"FFJ lam={lam}", "ff", "joint", lam=lam))
            mk.append(lambda rho, lam=lam: Policy(f"FF-JR lam={lam}", "ff_rate", "joint", lam=lam))
            mk.append(lambda rho, lam=lam: Policy(f"FF lam={lam}", "ff", "thresh", lam=lam))
        grid("E2", mk + ["J-EFF", "EFF", "SEQ"], [("default", {})], seeds=[0])
    if which in ("E3", "all"):  # energy-access scarcity
        cfgs = [(f"cpc={v}", dict(couriers_per_cab=v, km2_per_cab=99.0)) for v in [4, 6, 8, 12, 16]]
        cfgs += [(f"slots={v}", dict(cab_slots=v)) for v in [6, 8, 14]]
        cfgs += [(f"bshare={v}", dict(b_cab_share=v)) for v in [0.2, 0.6, 1.0]]
        cfgs += [(f"ext={v}", dict(ext_rate=v)) for v in [0.0, 4.0]]
        grid("E3", CORE, cfgs, seeds=[0, 1])
    if which in ("E4", "all"):  # paired heterogeneity intervention
        grid("E4", CORE + NOCTG, [("hetero", {}), ("homog", dict(share_heavy=0.0, share_b=0.0)),
                          ("heavy-only", dict(share_b=0.0)), ("typeB-only", dict(share_heavy=0.0))])
    if which in ("E5", "all"):  # infeasible / ambitious floors
        for k in [0.8, 1.0, 1.1]:
            grid(f"E5_k{k}", ["EFF", "FFJ", "J-EFF", "FF-JR", "OURS"], [(f"kappa={k}", {})], seeds=[0, 1], rho_override=k)
    if which in ("E6", "all"):  # supply-demand shift with rho fixed from calibration
        grid("E6", CORE, [(f"fpo={v}", dict(fleet_per_orders=v)) for v in [20.0, 30.0, 50.0, 60.0]], seeds=[0, 1])
    if which in ("E7", "all"):  # energy-model error and reserve
        cfgs = [(f"noise={e},res={r}", dict(e_noise=e, reserve=r)) for e in [0.1, 0.2, 0.3] for r in [0.05, 0.10]]
        grid("E7", ["AGN", "EFF", "J-EFF", "OURS"], cfgs, seeds=[0, 1])
    if which in ("E8", "all"):  # runtime vs fleet size (SH-5, one week)
        jobs = []
        for fpo in [40.0, 20.0, 10.0, 5.0]:
            cfg = replace(Cfg(), fleet_per_orders=fpo)
            w = weeks("SH-5")[1][0][:2]
            for name in ["EFF", "FFJ", "OURS"]:
                jobs.append(("E8", "SH-5", w, cfg, policies(sel["rho"]["SH-5"], sel["V"], sel["lam"])[name], sel["rho"]["SH-5"], f"fpo={fpo}"))
        with Pool(4) as p:  # few parallel jobs so timings are not distorted
            out = p.map(job, jobs, chunksize=1)
        pd.DataFrame([o for o, _, _ in out]).to_csv(RES / "E8.csv", index=False)
    if which in ("E9", "all"):  # queue trajectories and the V trade-off
        mk = [lambda rho, V=V: Policy(f"OURS V={V}", "dpp", "joint", V=V, rho=rho) for V in [0.1, 1, 10, 100]]
        grid("E9", mk, [("default", {})], seeds=[0], n_weeks=2)
    if which in ("E12", "all"):  # theorem-form (linear) weights vs deployed renewal-ratio weights
        mk = [lambda rho, V=V: Policy(f"IFD-lin V={V}", "dpp_lin", "joint", V=V, rho=rho) for V in [0.1, 1, 10]]
        grid("E12", mk + ["OURS"], [("default", {})], seeds=[0, 1])
    if which in ("E13", "all"):  # sensitivity: larger cabinets; flat pay with floors recalibrated
        grid("E13a", CORE, [("pcab=2.4kW", dict(p_cab=2400.0)), ("slots=16", dict(cab_slots=16, p_cab=7200.0))], seeds=[0, 1])
        flat = dict(pay_base=3.5, pay_km=0.0)
        jobs = []
        for d in DISTS:
            for cal in weeks(d)[0]:
                for s in SEEDS:
                    jobs.append(("cal", d, cal, replace(Cfg(), seed=s, **flat), Policy("J-EFF", "dpp", "joint", V=1, rho=0.0), None, "flat"))
        c = pool_run(jobs, "E13_cal")
        sel = load_cal()
        jobs = []
        for d in DISTS:
            rho = KAPPA * c[c.dist == d].rate_mean.mean()
            P = policies(rho, sel["V"], sel["lam"], sel.get("lamr"))
            for w in weeks(d)[1]:
                for s in [0, 1]:
                    for name in CORE:
                        jobs.append(("E13b", d, w, replace(Cfg(), seed=s, **flat), P[name], rho, "flat-pay"))
        pool_run(jobs, "E13b")
    if which in ("E14", "all"):  # four-week horizon: queue stability and floor attainment
        from week import inst
        jobs = []
        for d in DISTS:
            days = sorted(inst(d)["days"])[7:35]
            P = policies(sel["rho"][d], sel["V"], sel["lam"], sel.get("lamr"))
            for name in ["J-EFF", "FFJ", "FF-JR", "OURS"]:
                jobs.append(("E14", d, days, replace(Cfg(), seed=0), P[name], sel["rho"][d], "4wk"))
        with Pool(16) as p:
            out = p.map(job, [(e, d, w, c, pol, r, t) for (e, d, w, c, pol, r, t) in jobs], chunksize=1)
        pd.DataFrame([o for o, _, _ in out]).to_csv(RES / "E14.csv", index=False)
        pd.concat([r for _, r, _ in out]).to_csv(RES / "E14_rates.csv", index=False)
        pd.concat([t for _, _, t in out if t is not None]).to_csv(RES / "E14_trace.csv", index=False)
    if which in ("E15", "all"):  # LaDe demand-construction sensitivity
        cfgs = [("lead=10", dict(lead=10)), ("lead=30", dict(lead=30)), ("window=30", dict(window=30)),
                ("window=60", dict(window=60)), ("shifts=random", dict(shift_pick="random")),
                ("booked", dict(lead=60, window=120))]
        grid("E15", CORE, cfgs, seeds=[0, 1])
    if which in ("E16", "all"):  # reservation externality: keep walk-in packs for private riders
        grid("E16", CORE, [(f"keep={v}", dict(res_keep=v)) for v in [1, 2]], seeds=[0, 1])
    if which in ("E1b", "all"):  # FairFoody baselines at lambda=1 (stronger fairness than the tie-break choice)
        mk = [lambda rho: Policy("FF lam=1", "ff", "thresh", lam=1.0), lambda rho: Policy("FFJ lam=1", "ff", "joint", lam=1.0)]
        grid("E1b", mk, [("default", {})])
    if which in ("E18", "all"):  # additional temporal validation: every whole 7-day block outside calibration and test weeks
        from week import inst
        jobs = []
        for d in DISTS:
            days = sorted(inst(d)["days"])
            blocks = [days[i:i + 7] for i in range(0, len(days) - 6, 7)]
            used = {tuple(b) for b in weeks(d)[0] + weeks(d)[1]}
            P = policies(sel["rho"][d], sel["V"], sel["lam"], sel.get("lamr"))
            for b in blocks:
                if tuple(b) in used:
                    continue
                for name in CORE + ["EFF-R", "SEQ-R", "J-EFF-0", "OURS-0"]:
                    for s in [0, 1]:
                        jobs.append(("E18", d, b, replace(Cfg(), seed=s), P[name], sel["rho"][d], "holdout"))
        pool_run(jobs, "E18")
    if which in ("E19", "all"):  # constrained cabinet network (the earlier default): 6 slots, 400 W, 1.6 kW cap
        grid("E19", CORE, [("constrained", dict(cab_slots=6, p_slot=400.0, p_cab=1600.0))], seeds=[0, 1, 2])
    if which in ("E20", "all"):  # private riders: do not yield to couriers / try another cabinet
        grid("E20", ["EFF", "J-EFF", "FF-JR", "OURS"], [("priv=equal", dict(ext_policy="equal")), ("priv=retry", dict(ext_policy="retry"))], seeds=[0, 1])
    if which in ("E21", "all"):  # absolute floor (23 RMB/h for every district): infeasible by construction under our pay model
        jobs = []
        for d in DISTS:
            P = policies(23.0, sel["V"], sel["lam"], sel.get("lamr"))
            for w in weeks(d)[1]:
                for s in [0, 1]:
                    for name in ["J-EFF", "FF-JR", "OURS"]:
                        jobs.append(("E21", d, w, replace(Cfg(), seed=s), P[name], 23.0, "rho=23"))
        pool_run(jobs, "E21")
    if which in ("E22", "all"):  # gaming: 20% of couriers pad online time by 60 min on each side
        grid("E22", ["J-EFF", "FF-JR", "OURS", "IFD-Q"], [("honest", dict(gamer_share=0.0)), ("gaming", dict(gamer_share=0.2))], seeds=[0, 1])
    if which in ("E23", "all"):  # overnight continuity: SoC and position carry over (no home charging)
        grid("E23", CORE + NOCTG, [("hetero", dict(carry=1.0)), ("homog", dict(carry=1.0, share_heavy=0.0, share_b=0.0))], seeds=[0, 1])
    if which in ("E24", "all"):  # swap-weight knobs that Table S14 leaves fixed
        knobs = [("kappa_s=0.5", dict(kappa_s=0.5)), ("kappa_s=2", dict(kappa_s=2.0)), ("th_pro=0.35", dict(th_pro=0.35)),
                 ("th_pro=0.55", dict(th_pro=0.55)), ("c_wait=0.1", dict(c_wait=0.1)), ("c_wait=0.4", dict(c_wait=0.4)),
                 ("n_res=1", dict(n_res=1)), ("n_res=5", dict(n_res=5)), ("res_slack=5", dict(res_slack=5)), ("res_slack=20", dict(res_slack=20))]
        grid("E24", ["J-EFF", "FF-JR", "OURS"], knobs, seeds=[0])
        grid("E24c", ["J-EFF", "FF-JR", "OURS"], [(f"constrained {t}", dict(cab_slots=6, p_slot=400, p_cab=1600, **kw)) for t, kw in knobs], seeds=[0])
    if which in ("E25", "all"):  # efficiency-only joint matching with charger columns, not rate-normalized
        grid("E25", ["EFF-J"], [("default", {})])
        grid("E25c", ["EFF-J"], [("constrained", dict(cab_slots=6, p_slot=400.0, p_cab=1600.0))])
    if which in ("E27", "all"):  # strategic padding: online-time vs credited-time accounting, honest vs 20% gamers
        pols = ["J-EFF", "OURS", "OURS-C", "FF-JR", "FF-JR-C"]
        grid("E27", pols, [(tag, dict(gamer_share=0.2, gaming=tag)) for tag in ("none", "inert", "extend")], seeds=[0, 1])
    if which in ("E28", "all"):  # per-courier net income and FF-JR at lambda=10, same seeds and weeks as E1
        grid("E28", ["J-EFF", "J-EFF-0", "OURS", "OURS-0", "FF-JR", "FF-JR-10"], [("default", {})])
    if which in ("E29", "all"):  # one-factor contrasts in the constrained network
        grid("E29", ["EFF-R", "EFF-R0", "SEQ-R", "SEQ-R-noLock", "J-EFF-noLock", "J-EFF-0", "OURS-0", "OURS-noQswap", "FF-JR-10"],
             [("constrained", dict(cab_slots=6, p_slot=400.0, p_cab=1600.0))], seeds=[0, 1, 2])
    if which in ("E10", "all"):  # cabinet placement
        grid("E10", CORE, [("random-sites", dict(cab_place="random"))], seeds=[0, 1])


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "all")
