Set-Location "C:\Reinforcement-trading"

# This box also runs Wit-Hedge-fund's live paper-trading bot. Idle priority
# means Windows only schedules this process's threads when the bot (and
# everything else) isn't asking for CPU -- belt-and-suspenders on top of the
# OMP/MKL/torch single-thread caps set in run_vps_training.py itself.
try { (Get-Process -Id $PID).PriorityClass = 'Idle' } catch {}

& ".venv\Scripts\python.exe" "-u" "run_vps_training.py" 2>&1 |
    Out-File -FilePath "C:\Reinforcement-trading\outputs\rl_train_log.txt" -Append -Encoding utf8
