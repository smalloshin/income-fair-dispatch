"""Build simulation instances (district x day) from LaDe-P pickup records.

Each instance holds the realized pickup requests of one district-day (time, location)
and the real shift spans of that day's most active couriers. Courier positions and
assignments are NOT replayed: the simulator re-dispatches everything.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "lade"
OUT = ROOT / "data" / "instances"

# district -> (city file, region ids). Jilin core merges five adjacent small regions.
DISTRICTS = {
    "SH-5": ("sh", [5]),
    "SH-43": ("sh", [43]),
    "YT-111": ("yt", [111]),
    "YT-125": ("yt", [125]),
    "JL-core": ("jl", [109, 110, 122, 128, 131]),
}
T_START, T_END = 7 * 60, 19 * 60  # pickups kept if realized in [07:00, 19:00)


def minutes(s):
    t = pd.to_datetime("2022-" + s, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    return t.dt.hour * 60 + t.dt.minute


def build(name, city, regions):
    df = pd.read_csv(RAW / f"pickup_{city}.csv",
                     usecols=["order_id", "region_id", "courier_id", "pickup_time", "accept_time", "lng", "lat", "ds"])
    df = df[df.region_id.isin(regions)].copy()
    lat0, lng0 = df.lat.median(), df.lng.median()
    df["x"] = (df.lng - lng0) * 111.32 * np.cos(np.radians(lat0))
    df["y"] = (df.lat - lat0) * 110.57
    df = df[np.hypot(df.x, df.y) <= np.percentile(np.hypot(df.x, df.y), 99)]  # drop far outliers
    df["tp"] = minutes(df.pickup_time)
    df = df[(df.tp >= T_START) & (df.tp < T_END)]
    hub = (float(df.x.median()), float(df.y.median()))
    days = {}
    for ds, d in df.groupby("ds"):
        d = d.sort_values("tp")
        span = d.groupby("courier_id").agg(n=("order_id", "size"), s=("tp", "min"), e=("tp", "max"))
        days[str(ds).zfill(4)] = {
            "t": d.tp.to_numpy(np.int32), "x": d.x.to_numpy(), "y": d.y.to_numpy(),
            # real couriers of the day, most active first: (tasks, first pickup, last pickup)
            "shifts": span.sort_values("n", ascending=False)[["n", "s", "e"]].to_numpy(np.int32),
        }
    n = np.array([len(v["t"]) for v in days.values()])
    lo, hi = np.percentile(n, [10, 90])
    keep = {k: v for k, v in days.items() if lo <= len(v["t"]) <= hi}  # drop atypical days
    inst = {"name": name, "hub": hub, "lat0": lat0, "lng0": lng0, "days": keep,
            "bbox": (df.x.min(), df.x.max(), df.y.min(), df.y.max())}
    print(name, "days", len(keep), "orders/day %.0f" % np.mean([len(v['t']) for v in keep.values()]),
          "real couriers/day %.1f" % np.mean([len(v['shifts']) for v in keep.values()]))
    return inst


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name in (sys.argv[1:] or DISTRICTS):
        city, regs = DISTRICTS[name]
        with open(OUT / f"{name}.pkl", "wb") as f:
            pickle.dump(build(name, city, regs), f)
