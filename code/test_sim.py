"""Self-checks for the simulator invariants claimed in the paper. Run: python3 -I code/test_sim.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from sim import NEG, Cfg, Policy, Sim  # noqa: E402
from week import inst, run_week, weeks  # noqa: E402


def test_invariants():
    cfg = Cfg(seed=1)
    cal, test = weeks("SH-43")
    for pol in [Policy("EFF", "eff", "thresh"), Policy("SEQ", "eff", "seq"), Policy("FFJ", "ff", "joint"),
                Policy("OURS", "dpp", "joint", V=1, rho=14.0), Policy("NORES", "dpp", "joint_nores", V=1, rho=14.0)]:
        o, x = run_week("SH-43", test[0], cfg, pol)  # run_week asserts the deficit inequality per day
        assert o["strand"] == 0, (pol.name, o["strand"])  # zero stranding with exact energy model
        assert np.all(x["Q"] >= 0)
    o, _ = run_week("SH-43", test[0], cfg, Policy("AGN", "eff", "thresh", energy_filter=False))
    print("battery-agnostic strandings in one week:", o["strand"])


def test_conservation_and_matching():
    I = inst("YT-125")
    d = sorted(I["days"])[3]
    s = Sim(I, d, Cfg(), Policy("OURS", "dpp", "joint", V=1, rho=15.0))
    n0 = sum(len(e) for e in s.pk_E)
    s.run()  # run() asserts pack conservation at the end
    assert sum(len(e) for e in s.pk_E) == n0
    # all-negative weights must leave every courier idle (dummy columns)
    s2 = Sim(I, d, Cfg(), Policy("EFF", "eff", "thresh"))
    W = np.full((3, 4), NEG)
    W[0, 1] = -5.0
    assert s2.match(0, np.arange(3), np.arange(4), W, []) == set()


def test_deficit_inequality_by_hand():
    rho, q, earned, online = 12.0, 0.0, 0.0, 0
    rng = np.random.default_rng(0)
    for _ in range(600):
        a = rng.random() < 0.9
        r = 4.0 if rng.random() < 0.05 else 0.0
        q = max(q + rho / 60 * a - r, 0.0)
        earned += r
        online += a
        assert earned + 1e-9 >= rho * online / 60 - q


def test_policy_knobs():
    """Regression guard: every experiment must pass the calibrated FF-JR lambda (bug found in integrity round 2)."""
    from run_experiments import policies
    P = policies(1.0, 0.03, 0.3, 10.0)
    assert P["FF-JR"].lam == 10.0 and P["FFJ"].lam == 0.3 and P["FF-JR-0"].lam == 10.0
    src = (Path(__file__).parent / "run_experiments.py").read_text()
    assert "policies(rho, sel[\"V\"], sel[\"lam\"])" not in src


def test_inert_padding_invariance():
    """Proposition 4: under credited time, inert padding leaves assignments, earnings, and physical states unchanged."""
    from week import run_week, weeks
    days = weeks("YT-125")[1][0][:2]
    pol = Policy("IFD-C", "dpp", "joint", V=0.03, rho=14.0, credit=True)
    h = [run_week("YT-125", days, Cfg(seed=0, gamer_share=0.5, gaming=g), pol, rho_eval=14.0)[0]["path_hash"] for g in ("none", "inert")]
    assert h[0] == h[1]


if __name__ == "__main__":
    test_policy_knobs()
    test_deficit_inequality_by_hand()
    test_conservation_and_matching()
    test_invariants()
    test_inert_padding_invariance()
    print("all checks passed")

