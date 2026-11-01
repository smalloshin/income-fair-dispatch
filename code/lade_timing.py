"""Timing facts of LaDe pickups in the five districts (raw, before trimming): lead from acceptance to pickup, window length, pickup inside window. Usage: python3 -I code/lade_timing.py data/raw/lade"""
import sys
import pandas as pd
D = {"SH-5": ("sh", [5]), "SH-43": ("sh", [43]), "YT-111": ("yt", [111]), "YT-125": ("yt", [125]), "JL-core": ("jl", [109, 110, 122, 128, 131])}
raw = sys.argv[1]
rows = []
for name, (city, regs) in D.items():
    df = pd.read_csv(f"{raw}/pickup_{city}.csv", usecols=["region_id", "accept_time", "time_window_start", "time_window_end", "pickup_time"])
    df = df[df.region_id.isin(regs)]
    f = lambda c: pd.to_datetime("2022-" + df[c].astype(str), format="%Y-%m-%d %H:%M:%S", errors="coerce")
    acc, ws, we, pk = f("accept_time"), f("time_window_start"), f("time_window_end"), f("pickup_time")
    lead = (pk - acc).dt.total_seconds() / 60
    win = (we - ws).dt.total_seconds() / 60
    inside = (pk >= ws) & (pk <= we)
    rows.append(dict(district=name, n=len(df), lead_median=lead.median(), lead_ge60=(lead >= 60).mean(), win_median=win.median(),
                     win_120=(win == 120).mean(), inside=inside.mean()))
out = pd.DataFrame(rows)
print(out.round(3).to_string())
print("pooled n", out.n.sum())
