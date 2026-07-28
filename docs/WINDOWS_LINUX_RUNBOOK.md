# Running the RL repo on Windows and Linux

`inno-wit/Reinforcement-trading` — XAUUSD M1 → PPO bracket-trading pipeline (Gymnasium + Stable-Baselines3).

A working split: Windows for interactive iteration, Linux for long unattended training.

## What it is

Plain-Python research pipeline, nothing OS-specific in the trading logic:

`data_loader.py` → `features.py` → `env_bracket.py` (the Gymnasium environment) → `train_ppo.py`
(PPO via Stable-Baselines3, walk-forward validated) → `final_holdout_eval.py`. No Docker, no CI.
Everything reads through `pathlib.Path` with forward-slash relative paths, so nothing needs
rewriting to move between the two OSes.

Two things the repo already gets right, so you don't have to touch them:

- `train_ppo.py:362` — picks the `SubprocVecEnv` start method per platform: `spawn` on `win32`,
  `forkserver` on Linux.
- `device="auto"` gates on `torch.cuda.is_available()` — the same code path lands on GPU or CPU
  on either OS without a branch.

## Gaps (status as of 2026-07-29)

1. **No `.gitignore`, ~23 years of M1 data** — FIXED (commit `2dce44a`). `.gitignore` now covers
   `data/*.csv`, `models/`, `outputs/`, `__pycache__/`, `.venv/`, `.ipynb_checkpoints/`. Verified
   nothing was previously tracked under those paths (`git ls-files` came back empty).
2. **Undocumented Python floor (3.10+)** — the concern was bare `str | None` union syntax in
   `training_diagnostics.py` / `view_results.py` without `from __future__ import annotations`.
   Checked on clone: **both files already have the future import** (lines 15 and 30), so this
   isn't a live bug. `.python-version` pinned to `3.11` anyway (commit `2dce44a`) since
   Gymnasium/SB3 benefit from it and it matches the setup commands below.

## Role split

| | Windows (daily driver) | Linux (VPS or WSL2) |
|---|---|---|
| Use for | Interactive iteration — `notebooks/`, `view_results.py`, `visualize.py`, quick baseline tuning | Long unattended PPO training — `train_ppo.py`, the ~35-fold sliding walk-forward over the full dataset |
| Why | You're already here; Plotly notebooks are nicer to work in locally | `forkserver` + no desktop overhead scales `n_envs` cleaner; runs multi-hour jobs without tying up your machine |

Linux side runs on a VPS, reusing the same SSH keypair pattern already set up for the
wit-nautilus VPS rather than provisioning a new one.

## Windows setup — interactive iteration

```powershell
git clone https://github.com/inno-wit/Reinforcement-trading
cd Reinforcement-trading
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**If you have an NVIDIA GPU:** CPU-only PyTorch installs automatically via `requirements.txt`
(Stable-Baselines3 pulls it in transitively). For CUDA, install torch explicitly first so the
CUDA wheel wins:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

**Data + smoke test:** drop the CSV into `data/` — it isn't in git, copy it manually or via a
synced drive.

```powershell
python run_pipeline.py
```

## Linux setup — unattended training (VPS or WSL2)

```bash
git clone https://github.com/inno-wit/Reinforcement-trading
cd Reinforcement-trading
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Same CUDA-first trick if the box has a GPU:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

No GPU (typical small VPS) → CPU-only torch installs automatically, which is fine —
`SubprocVecEnv` with `n_envs=4+` on a multi-core box is the intended fast path per the code's own
comments (`train_ppo.py:320-324`).

**Multi-hour job, no monitor attached** — run training under `tmux` so it survives an SSH
disconnect:

```bash
tmux new -s rl-train
python train_ppo.py
# Ctrl+B, D to detach
tmux attach -t rl-train   # check back in later
```

## Moving data and results between the two

Nothing in the repo syncs this for you. Reuse the same SSH keypair pattern already set up for
the wit-nautilus VPS (`docs/VPS_DEPLOYMENT.md` in that repo, step 1) rather than inventing a new
one.

| Direction | What |
|---|---|
| Windows → Linux, once, static | The historical CSV(s) under `data/` |
| Linux → Windows, after each training run | `models/` (checkpoints + `run_info.json`), `outputs/` (equity curves, `walk_forward_baseline.csv`) |

Push-button sync (e.g. a small `rsync` wrapper) can be added later if manual copying gets
tedious.

## Order of operations

1. ~~Add the `.gitignore` + Python version pin~~ — done, commit `2dce44a`.
2. **[Windows]** Stand up the Windows env — confirm `run_pipeline.py` and a short notebook run
   work against the short smoke-test CSV.
3. **[Linux]** Stand up the Linux side — the existing wit-nautilus VPS, or WSL2 to stay local
   first.
4. Push the full 23-year CSV to Linux, start training (`python train_ppo.py` under `tmux`).
5. Pull results back (`models/` + `outputs/`), iterate on Windows (`view_results.py`,
   `visualize.py`, notebooks).
