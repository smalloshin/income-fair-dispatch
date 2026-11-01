"""Tables (LaTeX snippets) and figures (PDF) for the paper from results/*.csv.

Inference is clustered by district-week: seeds are averaged within each district-week (20 clusters), then
paired across policies; p-values are Holm-adjusted within each metric's family of comparisons against IFD.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG, TAB = ROOT / "results", ROOT / "paper" / "figs", ROOT / "paper" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6, "legend.fontsize": 7,
                     "figure.dpi": 200, "savefig.bbox": "tight", "pdf.fonttype": 42})
COL = {"IFD-Q": "#bcbd22", "EFF": "#7f7f7f", "SEQ": "#bcbd22", "FF": "#ff9896", "FFJ": "#d62728", "FF-JR": "#9467bd", "J-EFF": "#1f77b4",
       "OURS": "#2ca02c", "NEAR": "#8c564b", "AGN": "#e377c2", "DPP-THR": "#17becf", "OURS-noRes": "#98df8a",
       "EFF-R": "#c7c7c7", "SEQ-R": "#dbdb8d"}
ORDER = ["NEAR", "EFF", "AGN", "SEQ", "FF", "FFJ", "EFF-R", "SEQ-R", "J-EFF", "FF-JR", "DPP-THR", "OURS-noRes", "IFD-Q", "OURS"]
NAMES = {"EFF-R0": "EFF-R0", "SEQ-R-noLock": "SEQ-R-noLock", "J-EFF-noLock": "J-EFF-noLock", "OURS-noQswap": "IFD-noQswap", "J-EFF-0": "J-EFF-0", "OURS-0": "IFD-0", "FF-JR-0": "FF-JR-0", "NEAR": "NEAR", "EFF": "EFF", "AGN": "AGN", "SEQ": "SEQ", "FF": "FF", "FFJ": "FF-J", "EFF-R": "EFF-R",
         "SEQ-R": "SEQ-R", "J-EFF": "J-EFF", "FF-JR": "FF-JR", "DPP-THR": "DPP-THR", "OURS-noRes": "IFD-noRes",
         "OURS": "IFD"}
W1 = 3.5  # IEEE column width (in)
CL = ["dist", "week"]  # inference cluster
KAPPA = 0.9  # calibrated floor multiplier
DISTS = ["SH-5", "SH-43", "YT-111", "YT-125", "JL-core"]


def nm(p):
    return NAMES.get(p, p)


def ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def clus(df, col):
    """Per-policy cluster means (district-week), seeds averaged."""
    return df.groupby(["policy"] + CL)[col].mean()


def paired_diff(df, a, b, col):
    """Cluster-paired difference a-b: mean, 95% t half-width, Wilcoxon p (n = district-weeks)."""
    g = clus(df, col)
    d = (g.loc[a] - g.loc[b]).dropna()
    if len(d) < 5:
        return np.nan, np.nan, np.nan
    p = np.nan if np.allclose(d, 0) else stats.wilcoxon(d).pvalue
    return d.mean(), ci(d), p


def holm(ps):
    ps = np.asarray(ps, float)
    out = np.full(len(ps), np.nan)
    ok = ~np.isnan(ps)
    idx = np.argsort(ps[ok])
    m = ok.sum()
    adj = np.maximum.accumulate([(m - k) * ps[ok][i] for k, i in enumerate(idx)])
    tmp = np.empty(m)
    tmp[idx] = np.minimum(adj, 1.0)
    out[ok] = tmp
    return out


MAIN_COLS = [("ontime_rate", "On time (\\%)", 100, 1), ("mean_wait", "Wait", 1, 2), ("pickup_delay", "Pickup delay", 1, 1),
             ("km_per_order", "km/req.", 1, 2), ("swap_wait", "Swap wait", 1, 1), ("strand", "Strand.", 1, 1),
             ("gini", "Gini ($\\times10^2$)", 100, 2), ("floor_viol", "Below $\\rho$ (\\%)", 100, 1),
             ("shortfall", "Shortfall", 1, 3), ("gap_dis", "Gap", 1, 2), ("ext_lost_share", "Priv. lost (\\%)", 100, 1)]


def table_main():
    df = pd.read_csv(RES / "E1.csv")
    pols = [p for p in ORDER if p in set(df.policy)]
    marks = {}
    for c, *_ in MAIN_COLS:
        others = [p for p in pols if p != "OURS"]
        ps = [paired_diff(df, "OURS", p, c)[2] for p in others]
        for p, q in zip(others, holm(ps)):
            marks[(p, c)] = q == q and q < 0.05
    lines = []
    for p in pols:
        g = clus(df, [c for c, *_ in MAIN_COLS]).loc[p]
        cells = []
        for c, _, s, k in MAIN_COLS:
            m, h = g[c].mean() * s, ci(g[c]) * s
            cells.append(f"{m:.{k}f}{{\\scriptsize$\\pm${h:.{k}f}}}" + ("$^\\dagger$" if marks.get((p, c)) else ""))
        name = "\\textbf{IFD (ours)}" if p == "OURS" else nm(p)
        lines.append(name + " & " + " & ".join(cells) + " \\\\")
        if p in ("FF", "SEQ-R"):
            lines.append("\\midrule")
    head = " & ".join(n for _, n, _, _ in MAIN_COLS)
    (TAB / "main.tex").write_text("\\begin{tabular}{l" + "r" * len(MAIN_COLS) + "}\n\\toprule\nPolicy & " + head
                                  + " \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    # per-district table
    g = df.groupby(["dist", "policy"])[["ontime_rate", "mean_wait", "swap_wait", "gini", "floor_viol", "gap_dis"]].mean()
    rows = []
    for dist in DISTS:
        for p in ["EFF", "FFJ", "J-EFF", "FF-JR", "OURS"]:
            r = g.loc[(dist, p)]
            rows.append(f"{dist if p == 'EFF' else ''} & {nm(p)} & {100 * r.ontime_rate:.1f} & {r.mean_wait:.2f} & "
                        f"{r.swap_wait:.1f} & {100 * r.gini:.2f} & {100 * r.floor_viol:.1f} & {r.gap_dis:.2f} \\\\")
        rows.append("\\midrule" if dist != DISTS[-1] else "")
    (TAB / "district.tex").write_text(
        "\\begin{tabular}{llrrrrrr}\n\\toprule\nDistrict & Policy & On time (\\%) & Wait & Swap wait & Gini ($\\times10^2$) & Below $\\rho$ (\\%) & Gap \\\\\n\\midrule\n"
        + "\n".join(r for r in rows if r) + "\n\\bottomrule\n\\end{tabular}\n")
    # paired-difference table of IFD against the closest alternatives
    rows, rec = [], []
    for b in ["J-EFF", "FF-JR", "FFJ"]:
        cells = []
        for c, lab, s, k in [("ontime_rate", "", 100, 2), ("mean_wait", "", 1, 2), ("swap_wait", "", 1, 2),
                             ("gini", "", 100, 2), ("shortfall", "", 1, 3), ("gap_dis", "", 1, 2), ("ext_lost_share", "", 100, 2)]:
            m, h, pv = paired_diff(df, "OURS", b, c)
            rec.append(dict(vs=b, metric=c, diff=m, ci=h, p=pv))
            cells.append(f"{m * s:+.{k}f}{{\\scriptsize$\\pm${h * s:.{k}f}}}")
        rows.append(f"{nm(b)} & " + " & ".join(cells) + " \\\\")
    (TAB / "diffs.tex").write_text("\\begin{tabular}{lrrrrrrr}\n\\toprule\nIFD $-$ & On time (pt) & Wait & Swap wait & Gini ($\\times10^2$) & Shortfall & Gap & Priv. lost (pt) \\\\\n\\midrule\n"
                                   + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    pd.DataFrame(rec).to_csv(RES / "summary_E1_diffs.csv", index=False)
    summ = df.groupby("policy")[[c for c, *_ in MAIN_COLS] + ["service_rate", "rate_p10", "topup_per_courier_week", "swaps_per_courier_day", "overtime_min_per_courier_day",
                                                              "dec_ms_mean", "dec_ms_p99", "res_fail", "stockouts"]].mean()
    summ.to_csv(RES / "summary_E1.csv")
    return summ


def table_ablation():
    """One-factor contrasts: each row differs from its reference in a single design factor."""
    df = pd.read_csv(RES / "E1.csv")
    pairs = [("EFF", "EFF-R0", "rate normalization only"), ("EFF-R0", "EFF-R", "+ cost-to-go"), ("EFF", "SEQ", "+ sequential reservation"),
             ("SEQ-R-noLock", "SEQ-R", "+ lock (sequential)"), ("J-EFF-noLock", "J-EFF", "+ lock (joint)"),
             ("SEQ-R", "J-EFF", "sequential $\\to$ joint"), ("J-EFF", "J-EFF-0", "$-$ cost-to-go (J-EFF)"),
             ("FF-JR", "OURS-noQswap", "FairFoody gap $\\to$ deficit (requests)"), ("OURS-noQswap", "OURS", "+ deficit in swap weights")]
    cols = [("ontime_rate", 100, 2), ("mean_wait", 1, 2), ("swap_wait", 1, 2), ("gini", 100, 2), ("shortfall", 1, 3), ("gap_dis", 1, 2)]
    rec = [dict(ref=a, new=b, lab=lab, metric=c, scale=s_, k=k, **dict(zip(("diff", "ci", "p"), paired_diff(df, b, a, c))))
           for a, b, lab in pairs for c, s_, k in cols]
    rec = pd.DataFrame(rec)
    rec["p_holm"] = rec.groupby("metric").p.transform(lambda x: holm(x.values))  # family: the rows of one column
    rows, rows_ci = [], []
    for (a, b, lab), g in rec.groupby(["ref", "new", "lab"], sort=False):
        cells = [f"{r['diff'] * r.scale:+.{r.k}f}" + ("$^*$" if r.p_holm < 0.05 else "") for _, r in g.iterrows()]
        cells_ci = [f"{r['diff'] * r.scale:+.{r.k}f}{{\\scriptsize$\\pm${r.ci * r.scale:.{r.k}f}}}" for _, r in g.iterrows()]
        rows.append(f"{nm(a)} $\\to$ {nm(b)} & {lab} & " + " & ".join(cells) + " \\\\")
        rows_ci.append(f"{nm(a)} $\\to$ {nm(b)} & " + " & ".join(cells_ci) + " \\\\")
    (TAB / "ablation_ci.tex").write_text("\\begin{tabular}{lrrrrrr}\n\\toprule\nContrast & On time (pt) & Wait & Swap wait & Gini ($\\times10^2$) & Shortfall & Gap \\\\\n\\midrule\n"
                                         + "\n".join(rows_ci) + "\n\\bottomrule\n\\end{tabular}\n")
    (TAB / "ablation.tex").write_text("\\begin{tabular}{llrrrrrr}\n\\toprule\nContrast & Factor & On time (pt) & Wait & Swap wait & Gini ($\\times10^2$) & Shortfall & Gap \\\\\n\\midrule\n"
                                      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    rec.to_csv(RES / "summary_ablation.csv", index=False)
    return rec


def km_per_courier_week(e):
    """Kilometres ridden per courier-week: km per served request x served requests / fleet."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from week import inst
    n_req = {}
    for d in e.dist.unique():
        I = inst(d)
        days = sorted(I["days"])
        for i in range(0, len(days) - 6, 7):  # same 7-day blocks as week.weeks
            n_req[(d, int(days[i]))] = sum(len(I["days"][x]["t"]) for x in days[i:i + 7])
    req = np.array([n_req[(d, int(w))] for d, w in zip(e.dist, e.week)])
    return e.assign(km_cw=e.km_per_order * e.service_rate * req / e.fleet)


def table_whopays():
    """Who pays for the floor: change in weekly earnings rate by the courier's J-EFF earnings quintile (same couriers,
    seeds, shifts, demand), and the platform-cost comparison with efficiency-only dispatch plus a cash top-up."""
    r = pd.read_csv(RES / "E1_rates.csv")
    e1 = km_per_courier_week(pd.read_csv(RES / "E1.csv"))
    K = ["dist", "week", "seed", "courier"]
    base = r[r.policy == "J-EFF"].set_index(K)
    base = base.assign(q=base.groupby(["dist", "week", "seed"]).rate.transform(lambda x: pd.qcut(x.rank(method="first"), 5, labels=False)))
    rows, rec = [], []
    for p in ["OURS", "IFD-Q", "FF-JR"]:
        x = r[r.policy == p].set_index(K)
        dlt = (x.rate - base.rate.loc[x.index]).to_frame("d").assign(q=base.q.loc[x.index].values).reset_index()
        g = dlt.groupby(["q"] + CL).d.mean()
        cells = []
        for q in range(5):
            v = g.loc[q]
            rec.append(dict(policy=p, quintile=q + 1, mean=v.mean(), ci=ci(v)))
            cells.append(f"{v.mean():+.2f}{{\\scriptsize$\\pm${ci(v):.2f}}}")
        # platform top-up and couriers' extra riding cost vs J-EFF (c_km = 0.3 RMB/km on km actually ridden per courier-week)
        m = e1.groupby("policy")[["topup_per_courier_week"]].mean()
        ride = 0.3 * paired_diff(e1, p, "J-EFF", "km_cw")[0]
        rows.append(f"{nm(p)} & " + " & ".join(cells) + f" & {m.loc[p, 'topup_per_courier_week']:.1f} & {ride:+.1f} \\\\")
    m = e1.groupby("policy")["topup_per_courier_week"].mean()
    rows.insert(0, "J-EFF & \\multicolumn{5}{c}{(reference)} & " + f"{m.loc['J-EFF']:.1f} & 0.0 \\\\")
    (TAB / "whopays.tex").write_text("\\begin{tabular}{lrrrrrrr}\n\\toprule\n & \\multicolumn{5}{c}{Change in earnings rate by J-EFF quintile (RMB/h)} & Top-up & Riding \\\\\n Policy & Q1 (low) & Q2 & Q3 & Q4 & Q5 (high) & (RMB/cw) & (RMB/cw) \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    out = pd.DataFrame(rec)
    out.to_csv(RES / "summary_whopays.csv", index=False)
    return out


def fig_tradeoff():
    df = pd.read_csv(RES / "E2.csv")
    g = df.groupby("policy")[["mean_wait", "pickup_delay", "shortfall", "gini", "ontime_rate", "floor_viol"]].mean()
    fig, ax = plt.subplots(1, 2, figsize=(2 * W1, 2.0))
    fams = [("OURS V=", COL["OURS"], "o", "IFD ($V$ sweep)"), ("FF-JR lam=", COL["FF-JR"], "v", "FF-JR ($\\lambda$ sweep)"),
            ("FFJ lam=", COL["FFJ"], "s", "FF-J ($\\lambda$ sweep)"), ("FF lam=", COL["FF"], "^", "FF ($\\lambda$ sweep)")]
    for fam, col, mk, lab in fams:
        s = g[g.index.str.startswith(fam)].copy()
        s["k"] = [float(i.split("=")[1]) for i in s.index]
        s = s.sort_values("k")
        for a, y in zip(ax, ["shortfall", "gini"]):
            a.plot(s.pickup_delay, s[y] * (100 if y == "gini" else 1), "-" + mk, color=col, ms=3, lw=1, label=lab)
    for p in ["EFF", "SEQ", "J-EFF"]:
        for a, y in zip(ax, ["shortfall", "gini"]):
            a.plot(g.loc[p, "pickup_delay"], g.loc[p, y] * (100 if y == "gini" else 1), "D", color=COL[p], ms=4, label=p)
    ax[0].set_ylabel("Shortfall (RMB/h)")
    ax[1].set_ylabel("Gini ($\\times10^2$)")
    for a in ax:
        a.set_xlabel("Mean pickup delay, release to arrival (min)")
        a.grid(alpha=0.3, lw=0.4)
    ax[1].legend(ncol=2, frameon=False, fontsize=6)
    fig.savefig(FIG / "tradeoff.pdf")
    plt.close(fig)
    return g


def fig_rates():
    r = pd.read_csv(RES / "E1_rates.csv")
    cal = json.load(open(RES / "calibration.json"))
    r["norm"] = r.rate / r.dist.map(cal["rho"])
    r["dis"] = r.heavy | (r.ptype == 1)
    pols = ["EFF", "SEQ", "FFJ", "J-EFF", "FF-JR", "OURS"]
    fig, ax = plt.subplots(1, 2, figsize=(2 * W1, 1.9))
    for p in pols:
        x = np.sort(r[r.policy == p].norm)
        ax[0].plot(x, np.arange(1, len(x) + 1) / len(x), color=COL[p], lw=1, label=nm(p))
    ax[0].axvline(1.0, color="k", lw=0.6, ls="--")
    ax[0].set_xlim(0.85, 1.08)
    ax[0].set_ylim(0, 0.45)
    ax[0].set_xlabel("Weekly earnings rate / $\\rho$")
    ax[0].set_ylabel("ECDF (lower tail)")
    ax[0].legend(frameon=False, fontsize=6)
    # cluster CIs: mean per (district-week, group), then t-interval over clusters
    m = r[r.policy.isin(pols)].groupby(["policy", "dis"] + CL).norm.mean().groupby(["policy", "dis"]).agg(["mean", ci]).reset_index()
    xs = np.arange(len(pols))
    for k, (dis, lab) in enumerate([(False, "advantaged"), (True, "energy-disadvantaged")]):
        mm = m[m.dis == dis].set_index("policy").loc[pols]
        ax[1].bar(xs + (k - 0.5) * 0.38, mm["mean"], 0.38, yerr=mm["ci"], color=["#9ecae1", "#fdae6b"][k],
                  edgecolor="k", lw=0.4, label=lab, capsize=1.5)
    ax[1].set_xticks(xs, [nm(p) for p in pols], fontsize=6)
    ax[1].set_ylim(0.95, 1.16)
    ax[1].axhline(1.0, color="k", lw=0.6, ls="--")
    ax[1].set_ylabel("Mean earnings rate / $\\rho$")
    ax[1].legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.18))
    fig.savefig(FIG / "rates.pdf")
    plt.close(fig)


def fig_scarcity():
    df = pd.read_csv(RES / "E3.csv")
    d = df[df.tag.str.startswith("cpc=")].copy()
    d["cpc"] = d.tag.str.split("=").str[1].astype(float)
    pols = ["EFF", "SEQ", "FFJ", "J-EFF", "FF-JR", "OURS"]
    fig, ax = plt.subplots(1, 3, figsize=(2 * W1, 1.8))
    for p in pols:
        g = d[d.policy == p].groupby("cpc")[["gap_dis", "shortfall", "swap_wait"]].mean()
        for a, c in zip(ax, ["gap_dis", "shortfall", "swap_wait"]):
            a.plot(g.index, g[c], "-o", ms=2.5, lw=1, color=COL[p], label=nm(p))
    for a, t in zip(ax, ["Energy-access gap (RMB/h)", "Mean shortfall (RMB/h)", "Mean swap wait (min)"]):
        a.set_xlabel("Couriers per cabinet")
        a.set_ylabel(t)
        a.grid(alpha=0.3, lw=0.4)
    ax[2].legend(frameon=False, fontsize=6)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(FIG / "scarcity.pdf")
    plt.close(fig)
    rows = []
    g_all = df.groupby(["tag", "policy"])[["ontime_rate", "mean_wait", "swap_wait", "shortfall", "gap_dis", "ext_lost_share"]].mean()
    for tag in ["cpc=4", "cpc=8", "cpc=16", "slots=6", "slots=8", "slots=14", "bshare=0.2", "bshare=0.6", "bshare=1.0", "ext=0.0", "ext=4.0"]:
        if tag not in g_all.index.get_level_values(0):
            continue
        for p in ["J-EFF", "FF-JR", "OURS"]:
            r = g_all.loc[(tag, p)]
            rows.append(f"{tag if p == 'J-EFF' else ''} & {nm(p)} & {100 * r.ontime_rate:.1f} & {r.mean_wait:.2f} & {r.swap_wait:.1f} & {r.shortfall:.3f} & {r.gap_dis:.2f} & {100 * r.ext_lost_share:.1f} \\\\")
    (TAB / "scarcity.tex").write_text("\\begin{tabular}{llrrrrrr}\n\\toprule\nSetting & Policy & On time (\\%) & Wait & Swap wait & Shortfall & Gap & Priv. lost (\\%) \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    return g_all


def fig_paired():
    """Energy-induced loss: same couriers, shifts, seeds and disturbances; only vehicle attributes differ.
    Loss is averaged over disadvantaged couriers within each episode; intervals are over district-week clusters."""
    r = pd.read_csv(RES / "E4_rates.csv")
    K = ["dist", "week", "seed", "policy", "courier"]
    base = r[r.tag == "homog"].set_index(K).rate
    out = []
    pols = ["EFF", "SEQ", "FFJ", "J-EFF", "FF-JR", "OURS"]
    for tag in ["hetero", "heavy-only", "typeB-only"]:
        h = r[r.tag == tag].set_index(K)
        h = h.assign(loss=base.loc[h.index].values - h.rate.values, dis=(h.heavy | (h.ptype == 1)).values).reset_index()
        for p in pols:
            e = h[(h.policy == p)]
            cd = e[e.dis].groupby(CL).loss.mean()
            ca = e[~e.dis].groupby(CL).loss.mean()
            out.append(dict(tag=tag, policy=p, loss_dis=cd.mean(), ci_dis=ci(cd), loss_adv=ca.mean(), ci_adv=ci(ca),
                            n_courier_weeks=int(e.dis.sum()), n_clusters=len(cd)))
    o = pd.DataFrame(out)
    o.to_csv(RES / "summary_E4.csv", index=False)
    fig, ax = plt.subplots(figsize=(W1, 2.0))
    xs = np.arange(len(pols))
    for k, tag in enumerate(["heavy-only", "typeB-only", "hetero"]):
        mm = o[o.tag == tag].set_index("policy").loc[pols]
        ax.bar(xs + (k - 1) * 0.27, mm.loss_dis, 0.27, yerr=mm.ci_dis, capsize=1.5, edgecolor="k", lw=0.4,
               color=["#fdd0a2", "#fd8d3c", "#a63603"][k],
               label={"heavy-only": "heavy vehicle only", "typeB-only": "pack type B only", "hetero": "both (default)"}[tag])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(xs, [nm(p) for p in pols], fontsize=6.5)
    ax.set_ylabel("Energy-induced loss (RMB/h)")
    ax.legend(frameon=False, fontsize=6)
    fig.savefig(FIG / "paired.pdf")
    plt.close(fig)
    return o


def fig_queue():
    tr = pd.read_csv(RES / "E14_trace.csv") if (RES / "E14_trace.csv").exists() else None
    df = pd.read_csv(RES / "E9.csv")
    df["V"] = df.policy.str.split("=").str[1].astype(float)
    fig, ax = plt.subplots(1, 2, figsize=(2 * W1, 1.8))
    if tr is not None:
        one = tr[(tr.dist == "YT-111") & (tr.policy == "OURS")].reset_index(drop=True)
        dayidx = one.groupby("day", sort=False).ngroup()
        ax[0].plot(np.arange(len(one)) / 6.0, one.q_max, color="#006d2c", lw=0.8, label="max $Q_i$")
        ax[0].plot(np.arange(len(one)) / 6.0, one.q_mean, color="#74c476", lw=0.8, label="mean $Q_i$")
        ax[0].set_xlabel("Simulated online hours (28 working days, YT-111)")
        ax[0].legend(frameon=False, fontsize=6, loc="upper right")
    ax[0].set_ylabel("Deficit queue (RMB)")
    g = df.groupby("V")[["q_end_mean", "q_end_max", "ontime_rate", "mean_wait", "pickup_delay", "shortfall"]].mean()
    ax[1].plot(g.index, g.q_end_max, "-o", ms=3, color="#006d2c", label="max $Q_i$ at week end")
    ax[1].plot(g.index, g.q_end_mean, "-s", ms=3, color="#74c476", label="mean $Q_i$ at week end")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("V")
    ax[1].set_ylabel("Queue (RMB)")
    a2 = ax[1].twinx()
    a2.plot(g.index, g.pickup_delay, "--^", ms=3, color="#1f77b4", label="pickup delay (min)")
    a2.set_ylabel("Pickup delay (min)", color="#1f77b4")
    h1, l1 = ax[1].get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, frameon=False, fontsize=6, loc="upper left")
    fig.tight_layout(w_pad=0.8)
    fig.savefig(FIG / "queue.pdf")
    plt.close(fig)
    g.to_csv(RES / "summary_E9.csv")
    return g


def _rows(df, tags, pols, cols, fmt):
    g = df.groupby(["tag", "policy"])[[c for c, *_ in cols]].mean()
    rows = []
    for tag in tags:
        for p in pols:
            if (tag, p) not in g.index:
                continue
            r = g.loc[(tag, p)]
            rows.append(f"{fmt(tag) if p == pols[0] else ''} & {nm(p)} & " + " & ".join(f"{r[c] * s:.{k}f}" for c, s, k in cols) + " \\\\")
    return rows


def tables_misc():
    out = {}
    # floors
    frames = []
    for k in [0.8, 1.0, 1.1]:
        frames.append(pd.read_csv(RES / f"E5_k{k}.csv").assign(tag=f"{k}"))
    e1 = pd.read_csv(RES / "E1.csv")
    frames.append(e1[e1.seed.isin([0, 1])].assign(tag="0.9"))
    df = pd.concat(frames)
    cols = [("ontime_rate", 100, 1), ("mean_wait", 1, 2), ("floor_viol", 100, 1), ("shortfall", 1, 3), ("topup_per_courier_week", 1, 1), ("floor_infeasible_cert", 100, 0)]
    rows = _rows(df, ["0.8", "0.9", "1.0", "1.1"], ["EFF", "J-EFF", "FFJ", "FF-JR", "OURS"], cols, lambda t: t)
    (TAB / "floors.tex").write_text("\\begin{tabular}{llrrrrrr}\n\\toprule\n$\\kappa$ & Policy & On time (\\%) & Wait & Below $\\rho$ (\\%) & Shortfall & Top-up & Cert. (\\%) \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    out["floors"] = df.groupby(["tag", "policy"])[[c for c, *_ in cols]].mean()
    # supply shift
    df = pd.concat([pd.read_csv(RES / "E6.csv"), e1[e1.seed.isin([0, 1]) & e1.policy.isin(["EFF", "SEQ", "FFJ", "J-EFF", "FF-JR", "OURS"])].assign(tag="fpo=40.0")])
    cols = [("ontime_rate", 100, 1), ("mean_wait", 1, 2), ("swap_wait", 1, 1), ("floor_viol", 100, 1), ("shortfall", 1, 3)]
    rows = _rows(df, ["fpo=20.0", "fpo=30.0", "fpo=40.0", "fpo=50.0", "fpo=60.0"], ["EFF", "J-EFF", "FF-JR", "OURS"], cols, lambda t: t.split("=")[1].rstrip("0").rstrip("."))
    (TAB / "supply.tex").write_text("\\begin{tabular}{llrrrrr}\n\\toprule\nReq./courier & Policy & On time (\\%) & Wait & Swap wait & Below $\\rho$ (\\%) & Shortfall \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    out["supply"] = df.groupby(["tag", "policy"])[[c for c, *_ in cols]].mean()
    # energy-model error
    df = pd.read_csv(RES / "E7.csv")
    g = df.groupby(["tag", "policy"])[["strand", "ontime_rate", "shortfall"]].mean().unstack("policy")
    g.to_csv(RES / "summary_E7.csv")
    rows = []
    for tag in g.index:
        e, r = tag.split(",")
        rows.append(f"{e.split('=')[1]} & {r.split('=')[1]} & " + " & ".join(f"{g.loc[tag, ('strand', p)]:.1f}" for p in ["AGN", "EFF", "J-EFF", "OURS"]) + " \\\\")
    (TAB / "robust.tex").write_text("\\begin{tabular}{rrrrrr}\n\\toprule\n$\\sigma_E$ & Reserve & AGN & EFF & J-EFF & IFD \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    out["E7"] = g
    # runtime
    df = pd.read_csv(RES / "E8.csv")
    rows = []
    for tag in ["fpo=40.0", "fpo=20.0", "fpo=10.0", "fpo=5.0"]:
        e = df[df.tag == tag]
        rows.append(f"{int(e.fleet.iloc[0])} & " + " & ".join(f"{e[e.policy == p].dec_ms_mean.iloc[0]:.2f} / {e[e.policy == p].dec_ms_p99.iloc[0]:.1f} / {e[e.policy == p].dec_ms_max.iloc[0]:.1f}" for p in ["EFF", "FFJ", "OURS"]) + " \\\\")
    (TAB / "runtime.tex").write_text("\\begin{tabular}{rrrr}\n\\toprule\nCouriers & EFF & FF-J & IFD \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    out["E8"] = df[["tag", "policy", "fleet", "dec_ms_mean", "dec_ms_p99", "dec_ms_max"]]
    # sensitivity tables: placement, cabinet size, flat pay, LaDe construction, walk-in buffer
    cols = [("ontime_rate", 100, 1), ("mean_wait", 1, 2), ("swap_wait", 1, 1), ("gini", 100, 2), ("shortfall", 1, 3), ("gap_dis", 1, 2), ("ext_lost_share", 100, 1)]
    frames = []
    for f, tag in [("E19", None), ("E10", None), ("E13a", None), ("E13b", None), ("E15", None), ("E16", None), ("E20", None)]:
        p = RES / f"{f}.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
    if (RES / "E23.csv").exists():  # overnight carry-over; its "homog" rows only serve the paired losses
        frames.append(pd.read_csv(RES / "E23.csv").query("tag == 'hetero'").assign(tag="carry-over"))
    if frames:
        df = pd.concat(frames)
        tags = [t for t in ["constrained", "pcab=2.4kW", "slots=16", "random-sites", "flat-pay", "lead=10", "lead=30", "window=30", "window=60", "booked", "shifts=random", "keep=1", "keep=2", "priv=retry", "carry-over"] if t in set(df.tag)]
        rows = sum((_rows(df, [t], ["EFF", "SEQ", "FFJ", "J-EFF", "FF-JR", "OURS"] if t == "flat-pay" else ["J-EFF", "FF-JR", "OURS"], cols, lambda t: t)
                    for t in tags), [])
        if (RES / "E1b.csv").exists():  # FairFoody baselines at lambda=1 under default conditions
            b = pd.read_csv(RES / "E1b.csv").groupby("policy")[[c for c, *_ in cols]].mean()
            for k, pol in enumerate(["FF lam=1", "FFJ lam=1"]):
                r = b.loc[pol]
                rows.append(f"{'$\\lambda=1$' if k == 0 else ''} & {pol.split()[0].replace('FFJ', 'FF-J')} & " + " & ".join(f"{r[c] * s:.{kk}f}" for c, s, kk in cols) + " \\\\")
        (TAB / "sensitivity.tex").write_text("\\begin{tabular}{llrrrrrrr}\n\\toprule\nSetting & Policy & On time (\\%) & Wait & Swap wait & Gini ($\\times10^2$) & Shortfall & Gap & Priv. lost (\\%) \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
        out["sens"] = df.groupby(["tag", "policy"])[[c for c, *_ in cols]].mean()
    # swap-weight knobs (E24 provisioned, E24c constrained): one row per setting
    if (RES / "E24.csv").exists() and (RES / "E24c.csv").exists():
        rows = []
        for f, base in (("E24", "E1"), ("E24c", "E19")):
            e = pd.read_csv(RES / f"{f}.csv")
            b = pd.read_csv(RES / f"{base}.csv")  # default settings, same seed as the knob runs
            b = b[(b.seed == 0) & b.policy.isin(["J-EFF", "FF-JR", "OURS"])].assign(tag=("constrained default" if f == "E24c" else "default"))
            e = pd.concat([b, e])
            for tag, g in e.groupby("tag", sort=False):
                m = g.groupby("policy")[["swap_wait", "gap_dis"]].mean()
                d, _, pv = paired_diff(g, "OURS", "J-EFF", "shortfall")
                net, knob = ("constr.", tag.split(" ", 1)[1]) if tag.startswith("constrained") else ("prov.", tag)
                knob = knob.replace("kappa_s", "$\\kappa_{\\mathrm{s}}$").replace("th_pro", "$\\theta_{\\mathrm{p}}$").replace("c_wait", "$c_{\\mathrm{w}}$").replace("n_res", "$n_{\\mathrm{res}}$").replace("res_slack", "$\\sigma_{\\mathrm{res}}$")
                rows.append(f"{net} & {knob} & {m.swap_wait['J-EFF']:.2f} & {m.swap_wait['OURS']:.2f} & {d:.3f} ({pv:.3f}) & {m.gap_dis['J-EFF']:.2f} & {m.gap_dis['OURS']:.2f} & {m.gap_dis['FF-JR']:.2f} \\\\")
        (TAB / "knobs.tex").write_text("\\begin{tabular}{llrrrrrr}\n\\toprule\nNetwork & Setting & \\multicolumn{2}{c}{Swap wait} & $\\Delta$Shortfall ($p$) & \\multicolumn{3}{c}{Gap} \\\\\n & & J-EFF & IFD & IFD$-$J-EFF & J-EFF & IFD & FF-JR \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    # four-week horizon
    p = RES / "E14.csv"
    if p.exists():
        out["E14"] = pd.read_csv(p).groupby("policy")[["ontime_rate", "mean_wait", "gini", "floor_viol", "shortfall", "gap_dis", "q_end_mean", "q_end_max"]].mean()
    # oracle: ratios against the incumbent and against the certified dual bound
    p = RES / "E11_oracle.csv"
    if p.exists():
        df = pd.read_csv(p)
        res = {}
        for q in ["EFF", "FFJ", "FF-JR", "J-EFF", "OURS"]:
            if f"{q}_util" not in df:
                continue
            res[q] = dict(util_vs_incumbent=(df[f"{q}_util"] / df.orc_eff_util).mean(),
                          util_vs_bound=(df[f"{q}_util"] / df.orc_eff_bound).mean(),
                          util_vs_bound_ci=ci(df[f"{q}_util"] / df.orc_eff_bound),
                          minrate_vs_incumbent=(df[f"{q}_min_rate"] / df.orc_fair_min_rate).mean(),
                          minrate_vs_bound=(df[f"{q}_min_rate"] / df.orc_fair_bound).mean(),
                          minrate_vs_bound_ci=ci(df[f"{q}_min_rate"] / df.orc_fair_bound))
        out["E11"] = pd.DataFrame(res).T
        out["E11"].to_csv(RES / "summary_E11.csv")
    return out


def text_extras():
    """In-text numbers of the limits and robustness paragraphs (E20-E27) that no table carries."""
    out = {}
    K = ["dist", "week", "seed", "policy", "courier"]
    if (RES / "E20.csv").exists():  # private riders: retry at the nearest other cabinet
        e = pd.read_csv(RES / "E20.csv")
        for tag, g in e.groupby("tag"):
            out[f"E20 {tag} ext_lost_share"] = g.groupby("policy").ext_lost_share.mean()
            out[f"E20 {tag} IFD-J-EFF"] = pd.DataFrame({c: paired_diff(g, "OURS", "J-EFF", c) for c in ("shortfall", "gini", "gap_dis")},
                                                       index=["diff", "ci", "p"])
    if (RES / "E21.csv").exists():  # absolute floor
        out["E21"] = pd.read_csv(RES / "E21.csv").groupby("policy")[["topup_per_courier_week", "rate_ub", "floor_infeasible_cert", "floor_viol"]].mean()
    if (RES / "E22_rates.csv").exists():  # gaming: same couriers, honest vs gaming
        r = pd.read_csv(RES / "E22_rates.csv")
        h, g = r[r.tag == "honest"].set_index(K), r[r.tag == "gaming"].set_index(K)
        g = g.assign(d=g.earn - h.loc[g.index].earn, dh=g.hours - h.loc[g.index].hours).reset_index()
        gain = {p: g[(g.policy == p) & g.gamer].groupby(CL).d.mean() for p in ("J-EFF", "FF-JR", "OURS", "IFD-Q")}
        rows = {p: dict(gamer_gain=gain[p].mean(), others=g[(g.policy == p) & ~g.gamer].groupby(CL).d.mean().mean(),
                        extra_vs_jeff=(gain[p] - gain["J-EFF"]).mean(),
                        p_extra=stats.wilcoxon(gain[p] - gain["J-EFF"]).pvalue if p != "J-EFF" else np.nan,
                        extra_hours=g[(g.policy == p) & g.gamer].groupby(CL).dh.mean().mean()) for p in gain}
        out["E22"] = pd.DataFrame(rows).T
        out["E22 topup"] = pd.read_csv(RES / "E22.csv").groupby(["tag", "policy"]).topup_per_courier_week.mean().unstack()
    if (RES / "E23.csv").exists():  # overnight carry-over
        e = pd.read_csv(RES / "E23.csv")
        e = e[e.tag == "hetero"]
        out["E23 swaps"] = e.groupby("policy")[["swaps_per_courier_day", "swap_wait"]].mean()
        out["E23 IFD-J-EFF"] = pd.DataFrame({c: paired_diff(e, "OURS", "J-EFF", c) for c in ("shortfall", "gini", "gap_dis", "ontime_rate")},
                                            index=["diff", "ci", "p"])
        r = pd.read_csv(RES / "E23_rates.csv")
        base = r[r.tag == "homog"].set_index(K).rate
        hh = r[r.tag == "hetero"].set_index(K)
        hh = hh.assign(loss=base.loc[hh.index].values - hh.rate.values, dis=(hh.heavy | (hh.ptype == 1)).values).reset_index()
        loss = {p: hh[(hh.policy == p) & hh.dis].groupby(CL).loss.mean() for p in ("EFF", "J-EFF", "J-EFF-0", "FF-JR", "OURS")}
        out["E23 loss"] = pd.Series({p: v.mean() for p, v in loss.items()})
        out["E23 J-EFF minus J-EFF-0 loss p"] = pd.Series({"diff": (loss["J-EFF"] - loss["J-EFF-0"]).mean(),
                                                         "p": stats.wilcoxon(loss["J-EFF"] - loss["J-EFF-0"]).pvalue})
    if (RES / "E18.csv").exists():  # confirmatory hold-out: IFD against each policy, Holm over policies per metric
        h = pd.read_csv(RES / "E18.csv")
        others = [p for p in h.policy.unique() if p != "OURS"]
        for m in ("gini", "shortfall", "gap_dis", "topup_per_courier_week", "ontime_rate", "rate_p90"):
            res = pd.DataFrame([dict(zip(("diff", "ci", "p"), paired_diff(h, "OURS", p, m))) for p in others], index=others)
            out[f"E18 IFD minus policy: {m}"] = res.assign(p_holm=holm(res.p.values))
        out["E18 means"] = h.groupby("policy")[["gap_dis", "topup_per_courier_week", "shortfall", "mean_wait"]].mean()
    rho = json.load(open(RES / "calibration.json"))["rho"]
    if (RES / "E27_rates.csv").exists():  # strategic padding: payoff = earnings + top-up under the policy's own accounting
        r = pd.read_csv(RES / "E27_rates.csv")
        r = r.assign(rho=r.dist.map(rho))
        r = r.assign(top_on=np.maximum(r.rho * r.hours - r.earn, 0), top_cr=np.maximum(r.rho * r.credit - r.earn, 0))
        base = r[r.tag == "none"].set_index(K)
        rows = []
        for tag in ("inert", "extend"):
            g = r[r.tag == tag].set_index(K)
            d = pd.DataFrame({"dE": g.earn - base.earn.loc[g.index], "dTon": g.top_on - base.top_on.loc[g.index],
                              "dTcr": g.top_cr - base.top_cr.loc[g.index], "gamer": g.gamer}).reset_index()
            for p in ("J-EFF", "OURS", "FF-JR", "OURS-C", "FF-JR-C"):
                for who, sub in (("gamer", d[(d.policy == p) & d.gamer]), ("other", d[(d.policy == p) & ~d.gamer])):
                    tot = sub.dE + (sub.dTcr if p.endswith("-C") else sub.dTon)
                    cl = tot.groupby([sub.dist, sub.week]).mean()
                    rows.append(dict(tag=tag, policy=p, who=who, d_earn=sub.dE.mean(), d_payoff=tot.mean(),
                                     p=stats.wilcoxon(cl).pvalue if (cl != 0).any() else 1.0))
        out["E27 padding gains (RMB per courier-week)"] = pd.DataFrame(rows)
        e = pd.read_csv(RES / "E27.csv")
        e = e[e.tag == "none"]
        out["E27 honest IFD-C minus IFD"] = pd.DataFrame({c: paired_diff(e, "OURS-C", "OURS", c) for c in ("shortfall", "gini", "ontime_rate")},
                                                        index=["diff", "ci", "p"])
        out["E27 credited share"] = e.groupby("policy").credit_share.mean()
    for f, kappa in (("E1_rates.csv", 0.9), ("E5_k1.0_rates.csv", 1.0)):  # Proposition 5: T = rho*A - Y + S, per episode
        if not (RES / f).exists():
            continue
        r = pd.read_csv(RES / f)
        r = r.assign(rho=r.dist.map(rho) * kappa / KAPPA, earn=r.rate * r.hours)
        r = r.assign(x=r.rho * r.hours - r.earn)
        g = r.groupby(["dist", "week", "seed", "policy"]).agg(T=("x", lambda x: np.maximum(x, 0).sum()), S=("x", lambda x: np.maximum(-x, 0).sum()),
                                                              gap=("x", "sum"), n=("x", "size")).reset_index()
        for c in ("T", "S", "gap"):
            g[c] = g[c] / g.n
        b = g[g.policy == "J-EFF"].set_index(["dist", "week", "seed"])
        res = {}
        for p in ("OURS", "FF-JR"):
            q = g[g.policy == p].set_index(["dist", "week", "seed"])
            ceil = np.minimum(b["T"], b["S"])
            res[p] = dict(T_jeff=b["T"].mean(), aggregate_deficit=np.maximum(b.gap, 0).mean(), ceiling=ceil.mean(),
                          saving=(b["T"] - q["T"]).mean(), realized_share=(b["T"] - q["T"]).sum() / ceil.sum())
        out[f"Prop. 5 decomposition, kappa={kappa}"] = pd.DataFrame(res).T
    for f in ("E25", "E25c"):  # EFF-J: joint matching without rate normalization
        if (RES / f"{f}.csv").exists():
            out[f] = pd.read_csv(RES / f"{f}.csv")[["ontime_rate", "mean_wait", "swap_wait", "shortfall", "gap_dis"]].mean()
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    print(table_main().round(3).to_string())
    print(pd.read_csv(RES / "summary_E1_diffs.csv").round(4).to_string())
    print(table_ablation().round(4).to_string())
    print(table_whopays().round(3).to_string())
    for f in (fig_tradeoff, fig_rates, fig_scarcity, fig_paired, fig_queue):
        try:
            r = f()
            if r is not None:
                print(f.__name__)
                print(r.round(3).to_string())
        except FileNotFoundError as e:
            print("skip", f.__name__, e)
    for k, v in tables_misc().items():
        print(k)
        print(v.round(3).to_string())
    for k, v in text_extras().items():
        print(k)
        print(v.round(4).to_string())
