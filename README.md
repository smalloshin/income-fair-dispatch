# Income-Fair Dispatch

Simulation code for the paper *Rate-Normalized Joint Dispatch and Battery-Swap Reservation for Income-Fair Electric Two-Wheeler Couriers* (C.-C. Hung).

| File | Purpose |
| --- | --- |
| `code/prep.py` | Builds district-day instances from LaDe pickup records |
| `code/sim.py` | Simulator and all dispatch policies |
| `code/week.py` | Weekly episodes |
| `code/run_experiments.py` | Calibration and all experiments |
| `code/oracle.py`, `code/oracle_frontier.py` | Small-instance hindsight MILP and price of fairness |
| `code/analyze.py` | Tables, figures, and in-text numbers |
| `code/lade_timing.py` | Timing statistics of the raw LaDe pickups |
| `code/test_sim.py` | Self-checks of the simulator invariants |

## Requirements

Python 3 with NumPy, SciPy, pandas, and Matplotlib. Developed with NumPy 2.1.3, SciPy 1.15.3, and pandas 2.2.3.

## Data

The demand traces are the pickup part of [LaDe](https://huggingface.co/datasets/Cainiao-AI/LaDe) (Apache-2.0). They are not stored in this repository. From the repository root:

```bash
mkdir -p data/raw/lade
for c in jl yt sh; do curl -L -o data/raw/lade/pickup_$c.csv https://huggingface.co/datasets/Cainiao-AI/LaDe/resolve/main/pickup/pickup_$c.csv; done
```

## Run

```bash
python3 -I code/prep.py
python3 -I code/test_sim.py
python3 -I code/run_experiments.py all
python3 -I code/oracle.py
python3 -I code/oracle_frontier.py
python3 -I code/analyze.py
```

`run_experiments.py` also takes a single stage, such as `cal` or `E1`; run `cal` before any experiment, because it writes `results/calibration.json`. Results go to `results/`. `analyze.py` writes tables to `paper/tables/` and figures to `paper/figs/`.

## License

The code is released under the [MIT License](LICENSE). The LaDe data keep their own Apache-2.0 license.
