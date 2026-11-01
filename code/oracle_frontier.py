"""Price of fairness on the hindsight-oracle instances, without tie-breaking artefacts.

For each E11 instance, with U* (max utility) and m* (max minimum rate) taken from E11_oracle.csv:
  s_F: max utility subject to min rate >= m* (1 - tol)      -> price of fairness 1 - U(s_F)/U*
  s_E: max min rate subject to utility >= U* (1 - tol)      -> min-rate loss of utility maximization 1 - m(s_E)/m*
  U*(0.9): max utility subject to min rate >= 0.9 m*        -> one interior frontier point
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from oracle import ROOT, Cfg, Sim, Policy, replace, small_instance, solve  # noqa: E402
from week import inst  # noqa: E402

TOL = 1e-3


def frontier(row):
    dist, w0, n, seed = row["dist"], int(row["w0"]), int(row["couriers"]), int(row["seed"])
    day = next(k for k in inst(dist)["days"] if int(k) == int(row["day"]))  # CSV stores the day key as an integer
    cfg = replace(Cfg(seed=seed, noise=0.0), soc0=(0.95, 0.95), load_cap=99, ext_rate=0.0)
    I, _, _ = small_instance(dist, day, w0, n, cfg)
    base = Sim(I, day, cfg, Policy("x"), fleet=n)
    U, m = float(row["orc_eff_util"]), float(row["orc_fair_min_rate"])
    out = dict(dist=dist, day=day, seed=seed, U_star=U, m_star=m)
    for tag, kw in [("F", dict(objective="eff", min_rate_floor=m * (1 - TOL))),
                    ("E", dict(objective="fair", util_floor=U * (1 - TOL))),
                    ("a90", dict(objective="eff", min_rate_floor=0.9 * m))]:
        r = solve(base, **kw)
        out.update({f"{tag}_{k}": v for k, v in (r or {}).items()})
    return out


if __name__ == "__main__":
    e11 = pd.read_csv(ROOT / "results" / "E11_oracle.csv")
    with Pool(8) as p:
        out = p.map(frontier, [r for _, r in e11.iterrows()], chunksize=1)
    df = pd.DataFrame(out)
    df.to_csv(ROOT / "results" / "E11_frontier.csv", index=False)
    print(df.round(3).to_string())
    print("price of fairness (1 - U/U*):")
    print((1 - df[["F_util", "a90_util"]].div(df.U_star, axis=0)).agg(["mean", "min", "max", "count"]).round(4))
