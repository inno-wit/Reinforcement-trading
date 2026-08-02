"""VPS-local training entrypoint for the real (non-smoke-test) MT5-sourced
dataset exported by export_mt5_csv.py.

Does NOT touch config.py's checked-in defaults, which are sized for the full
23-year Bid dataset and shared across all three environments (Windows local,
Linux VPS/Alpaca, Windows VPS/MT5). This box currently only has ~3.5 months
of real M1 history (MetaQuotes-Demo doesn't retain more for XAUUSD -- see
export_mt5_csv.py run log), so the walk-forward block scheme needs scaled-
down embargo/fold-count/timestep settings, applied here at runtime per
docs/MULTI_VPS_RUNBOOK.md's "Running on a small dataset" section.

Uses train_walk_forward() (block-fold scheme), not the __main__ default
train_sliding_walk_forward() -- the sliding scheme's default 5-year train
window per fold can't fit inside 3.5 months of data at all (zero folds).
"""
from __future__ import annotations

import os

# This VPS is a 4-logical-core box shared with Wit-Hedge-fund's live paper-
# trading bot. torch/MKL/OpenBLAS default to sizing their intra-op thread
# pool to ALL logical cores, in EVERY process -- with n_envs=4 that's 5
# processes (1 learner + 4 SubprocVecEnv workers) each trying to grab all 4
# cores, which pegged this box at 100% load and put the live bot at risk.
# Must be set before torch is imported (thread-pool size is fixed at import).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import torch  # noqa: E402  (must follow the env vars above)
torch.set_num_threads(1)

from config import CFG  # noqa: E402
from train_ppo import train_walk_forward  # noqa: E402

CFG.csv_path = "data/XAUUSD_1 Min_MT5_2026.04.15_2026.07.31.csv"

# ~107 days of M1 -> ~1,700-2,000 H1 decision bars. Default embargo=200/folds=5
# leaves blocks smaller than 2*embargo (data_loader.make_walk_forward_folds
# raises ValueError). Scaled per the runbook's documented 50-day example
# (embargo=10, folds=2), a little wider since this run has ~2x that history.
CFG.split_embargo_bars = 15
CFG.n_walk_forward_folds = 3

if __name__ == "__main__":
    train_walk_forward(
        total_timesteps=300_000,        # cut further: fewer envs + thread-capped means
        min_timesteps_per_fold=100_000, # slower per-step throughput, keep wall-clock bounded
        n_envs=2,                       # leave 2 of 4 cores free for the OS + the live bot
        device="auto",
    )
