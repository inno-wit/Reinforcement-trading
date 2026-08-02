# Frozen

**Date:** 2026-08-03

This repo is frozen. No new features, model architectures, symbols, or execution
work until the condition below is met. See the consolidation plan in
`wit-nautilus` (the active production repo) for full context — three parallel
trading systems were reviewed; this one has the best research hygiene of the
three (real leakage checks, walk-forward, sealed holdout discipline) but zero
execution path and no committed trained model.

## What's still allowed while frozen

| Allowed | Why |
|---|---|
| `export_mt5_csv.py` | Produces XAUUSD history needed if wit-nautilus's MT5 data bridge is built |
| `rl-train` scheduled task on the Windows VPS | To be unregistered entirely as part of this freeze (see below), not left running |

## What's not allowed

Everything else: no new features, no architecture changes, no new training runs
beyond what's already scheduled to stop, no live/paper execution work of any
kind (none exists today — keep it that way until the precondition below holds).

## `rl-train` VPS task

The `rl-train` Task Scheduler job on the shared Windows VPS (169.58.87.1) is
being unregistered as part of this freeze, not left idle. It was current only
while this repo was an active parallel effort; it isn't now.

## Condition that lifts this freeze

This repo does not get reconsidered until, at minimum:

1. wit-nautilus's deterministic chain has been through its own sealed
   validation gate (PASS or otherwise), **and**
2. This repo independently produces a positive sealed holdout on the *full*
   dataset (not the smoke-test CSV currently committed), with an actual
   trained model artifact in hand, **and**
3. wit-nautilus is trading the same instrument (XAUUSD) this repo targets.

Even then, the plan calls for absorbing this as an inference-only desk inside
wit-nautilus (`wit/desks/rl_signal.py`, ONNX-exported, no torch in the live
image) — not reviving this repo as a standalone system.
