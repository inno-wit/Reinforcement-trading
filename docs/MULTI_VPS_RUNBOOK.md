# Running the RL repo across three environments

`inno-wit/Reinforcement-trading` — XAUUSD M1 → PPO bracket-trading pipeline (Gymnasium + Stable-Baselines3).

Three environments, not a sequential hand-off:

- **Windows (local)** — interactive iteration, daily driver.
- **Linux VPS** (existing — wit-nautilus) — unattended PPO training on **Alpaca**-sourced data.
- **Windows VPS** (reached over SSH) — unattended PPO training on **MT5**-sourced data.

The two VPS boxes run independently and in parallel — separate walk-forward experiments compared
against each other by data source, not one feeding the other.

## What it is

Plain-Python research pipeline, nothing OS-specific in the trading logic:

`data_loader.py` → `features.py` → `env_bracket.py` (the Gymnasium environment) → `train_ppo.py`
(PPO via Stable-Baselines3, walk-forward validated) → `final_holdout_eval.py`. No Docker, no CI.
Everything reads through `pathlib.Path` with forward-slash relative paths, so nothing needs
rewriting to move between OSes.

Two things the repo already gets right, so you don't have to touch them:

- `train_ppo.py:362` — picks the `SubprocVecEnv` start method per platform: `spawn` on `win32`,
  `forkserver` on Linux. Both are live now (Windows VPS + Linux VPS), so this branch actually
  matters again.
- `device="auto"` gates on `torch.cuda.is_available()` — the same code path lands on GPU or CPU
  on any of the three machines without a branch.

**Open item — Alpaca data format:** `data_loader.py` expects MT4/MT5-style minute CSVs (column
`Time (EET)`, candle-open timestamps, broker/server tz — see `config.py:36-53`). Alpaca's API
returns bars in a different shape (UTC timestamps, different column names). There's no
Alpaca→pipeline adapter in this repo yet — needed before the Linux VPS can actually run
`train_ppo.py` against live-pulled Alpaca data. Until that adapter exists, feed the Linux VPS the
same CSV format as everywhere else.

## Gaps (status as of 2026-07-29)

1. **No `.gitignore`, ~23 years of M1 data** — FIXED (commit `2dce44a`). `.gitignore` now covers
   `data/*.csv`, `models/`, `outputs/`, `__pycache__/`, `.venv/`, `.ipynb_checkpoints/`. Verified
   nothing was previously tracked under those paths (`git ls-files` came back empty).
2. **Undocumented Python floor (3.10+)** — the concern was bare `str | None` union syntax in
   `training_diagnostics.py` / `view_results.py` without `from __future__ import annotations`.
   Checked on clone: **both files already have the future import** (lines 15 and 30), so this
   isn't a live bug. `.python-version` pinned to `3.11` anyway (commit `2dce44a`) since
   Gymnasium/SB3 benefit from it and it matches the setup commands below.
3. **Alpaca adapter missing** — see "What it is" above. Blocks live-data training on the Linux VPS
   until built.
4. **Windows VPS torch import crash (`WinError 1114` on `c10.dll`)** — FIXED. `pip install`
   reports success either way; torch's DLLs additionally need the VC++ runtime, which a fresh
   Windows Server image doesn't have. See the Windows VPS setup section below.
5. **Non-ASCII console output crashes unattended runs** — FIXED (`train_ppo.py`, `run_pipeline.py`
   now call `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` on start). Both scripts
   print arrows/checkmarks (best-fold markers, gate verdicts) that crash on the default `cp1252`/
   `cp437` console a Scheduled Task or SSH session gets. Belt-and-suspenders: also run
   `setx PYTHONIOENCODING utf-8` once per machine (see Windows VPS setup).
6. **Default `split_embargo_bars`/`n_walk_forward_folds` assume the full 23-year CSV** — not a bug,
   but undocumented: on a small ad-hoc pull (e.g. a smoke-test CSV), the default `embargo=200`
   makes `split_train_val_test` return an **empty** val/test slice, and `make_walk_forward_folds`
   raises a `ValueError` about block size. See "Running on a small dataset" below.

## Role split

| | Windows (local) | Linux VPS (Alpaca) | Windows VPS (MT5) |
|---|---|---|---|
| Use for | Interactive iteration — `notebooks/`, `view_results.py`, `visualize.py`, quick baseline tuning | Unattended PPO training, Alpaca-sourced data | Unattended PPO training, MT5-sourced data |
| Data source | Whatever CSV you drop in `data/` | Alpaca API (needs adapter — see Gaps) | MT5 broker export, already matches the CSV format the pipeline expects |
| Why | You're already here; Plotly notebooks are nicer to work in locally | `forkserver` + no desktop overhead scales `n_envs` cleanly; box already exists | Native MT5 terminal/API access — matches the broker environment used for eventual live trading |

Both VPS boxes are reached over SSH, reusing the same SSH keypair pattern already set up for the
wit-nautilus VPS rather than provisioning new ones per box.

## Windows setup — interactive iteration (local)

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

## Linux VPS setup — unattended training (Alpaca)

```bash
ssh <user>@<linux-vps-host>
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

**No GPU (typical small VPS):** unlike Windows, the default PyPI `torch` wheel on Linux x86_64 is
the CUDA build, not CPU-only — installing it drags in ~2.5 GB of `nvidia-*` CUDA runtime packages
that will never run, which can fail the install outright or eat the disk on a small VPS. Install
CPU torch explicitly first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

`SubprocVecEnv` with `n_envs=4+` on a multi-core box is the intended fast path per the code's own
comments (`train_ppo.py:320-324`).

Until the Alpaca adapter (see Gaps) exists, get data into `data/` the same way as everywhere else
— export/copy a CSV in the expected format, don't point `train_ppo.py` at a raw Alpaca pull yet.

**Multi-hour job, no monitor attached** — run training under `tmux` so it survives an SSH
disconnect:

```bash
tmux new -s rl-train
python train_ppo.py
# Ctrl+B, D to detach
tmux attach -t rl-train   # check back in later
```

## Windows VPS setup — unattended training (MT5)

**This box is shared with Wit-Hedge-fund** (a separate repo — an unattended paper-trading bot,
Task Scheduler jobs `wit-fund-schedule` / `wit-fund-watchdog` / `wit-fund-sshd`). Leave every
`wit-fund-*` task alone — `wit-fund-sshd` in particular is the only reason SSH stays up on this
VM image (its native sshd Windows Service is broken on this image), so removing it locks out
remote access entirely. Keep `n_envs` conservative and check core count before scaling up, and
prefix any Scheduled Task this repo creates with `rl-` so ownership is unambiguous.

```powershell
ssh <user>@<windows-vps-host>
git clone https://github.com/inno-wit/Reinforcement-trading
cd Reinforcement-trading
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Before `pip install` finishes anything useful, torch also needs the VC++ runtime** — a fresh
Windows Server image doesn't have it. Missing it surfaces as `import torch` raising
`OSError: [WinError 1114] ... c10.dll`, which names a file inside the torch package and reads
like a bad wheel, not a missing system runtime — install this first:

```powershell
winget install --id Microsoft.VCRedist.2015+.x64 -e --accept-package-agreements --accept-source-agreements
```

Same CUDA-first trick if the VPS has a GPU:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

No GPU (typical small VPS) → CPU-only torch installs automatically, same `SubprocVecEnv`
`n_envs=4+` fast path as the Linux side (`train_ppo.py:320-324`), just on the `spawn` start
method instead of `forkserver`.

MT5 exports already match the CSV format `data_loader.py` expects (see `README.md:3-4`), so no
adapter is needed here — drop the export into `data/` same as local.

**Multi-hour job, no session attached** — an SSH disconnect kills everything still tied to that
session's job object, same problem `tmux` solves on Linux. There's no `tmux` on Windows, so use a
Scheduled Task instead — it runs outside the SSH session's job object and survives disconnects.
`train_ppo.py`/`run_pipeline.py` already force UTF-8 stdout on start, but a Scheduled Task's
non-interactive console defaults to a codepage (`cp1252`/`cp437`) that can't print the
arrows/checkmarks these scripts log — set `PYTHONIOENCODING` once per machine as a second layer:

```powershell
setx PYTHONIOENCODING utf-8
schtasks /create /tn rl-train /tr "powershell -NoProfile -Command \"cd C:\path\to\Reinforcement-trading; .venv\Scripts\Activate.ps1; python train_ppo.py *> outputs\train_log.txt\"" /sc once /st 00:00 /f
schtasks /run /tn rl-train
```

Check on it from a later SSH session:

```powershell
schtasks /query /tn rl-train
Get-Content outputs\train_log.txt -Tail 50
```

Clean it up once training finishes:

```powershell
schtasks /delete /tn rl-train /f
```

## Running on a small dataset (smoke tests, ad-hoc pulls)

`config.py`'s `split_embargo_bars=200` and `n_walk_forward_folds=5` are sized for the full 23-year
CSV. On a small pull (e.g. a few weeks of M1 data for a plumbing check), the default embargo alone
can consume the entire val/test slice — `split_train_val_test` returns them **empty** rather than
erroring, and `make_walk_forward_folds` raises a `ValueError` about block size being too small.
Scale both down **at runtime**, not by editing the checked-in `config.py`:

```python
from config import CFG
CFG.split_embargo_bars = 10       # down from 200
CFG.n_walk_forward_folds = 2      # down from 5
```

`10`/`2` works for roughly 50 days of M1 data at the default `H1` decision timeframe. Treat the
resulting numbers as a plumbing check only (a handful of trades in a validation window says
nothing about profitability) — they exist to prove the environment and code run end-to-end, not to
produce a usable backtest.

## Moving data and results between the three

Nothing in the repo syncs this for you. Reuse the same SSH keypair pattern already set up for the
wit-nautilus VPS (`docs/VPS_DEPLOYMENT.md` in that repo, step 1) rather than inventing new ones —
`scp`/`rsync` for the Linux VPS, `scp` (OpenSSH ships it on Windows too) or `robocopy` for the
Windows VPS.

| Direction | What |
|---|---|
| Local → each VPS, once, static | The historical CSV(s) under `data/` (MT5 export for the Windows VPS; same format for the Linux VPS until the Alpaca adapter exists) |
| Each VPS → Local, after each training run | `models/` (checkpoints + `run_info.json`), `outputs/` (equity curves, `walk_forward_baseline.csv`) — pull separately per box, don't merge run directories, since Alpaca-trained and MT5-trained runs are different experiments |

Push-button sync (e.g. a small `rsync`/`scp` wrapper per box) can be added later if manual copying
gets tedious.

## Order of operations

1. ~~Add the `.gitignore` + Python version pin~~ — done, commit `2dce44a`.
2. **[Windows local]** Stand up the local env — confirm `run_pipeline.py` and a short notebook run
   work against the short smoke-test CSV.
3. **[Linux VPS]** Stand up the existing wit-nautilus box, if not already running. Build the
   Alpaca→pipeline adapter (Gaps #3) before pointing training at live Alpaca data.
4. **[Windows VPS]** SSH in and stand up the identical Windows env there.
5. Push data to each VPS (MT5 export to the Windows VPS, matching-format CSV to the Linux VPS),
   start training on both independently (`tmux` on Linux, Scheduled Task on Windows).
6. Pull results back from each box separately (`models/` + `outputs/`), iterate locally
   (`view_results.py`, `visualize.py`, notebooks) — compare Alpaca-run vs. MT5-run results rather
   than treating them as one dataset.
