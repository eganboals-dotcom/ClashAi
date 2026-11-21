# Runbook

## Check the setup

Use the project's Python executable with `scripts/lab.py doctor`, `coverage`, and `verify-model`. Reports go to ignored `local/` files. Hash verification and GPU matrix operations are not game-playing benchmarks.

## Evaluate strength

With the offline engine listening locally, supply a real upstream MatchConfig JSON for each matchup:

```powershell
.venv/Scripts/python.exe scripts/lab.py evaluate --match local/match.json --games 200
```

The default comparison is General versus the Hog specialist. Pass `--candidate` and `--opponent` to compare other trusted checkpoints. Upstream swaps sides on alternate games and records wins, draws, truncations, actions, and rejections. Test many deck pairs and tower troops, not only one favorable matchup. Review passive-opponent failures and timing parity. Never promote a checkpoint solely because it is newer or trained for more games.

## Monitor training

Read `runs/<name>/status.json` and `console-*.log`. Each attempt has its own upstream metrics, matches, league snapshots, and strict resume checkpoint. Report one attempt with:

```bash
.venv/bin/python scripts/lab.py report runs/week-1/attempt-00
```

The report counts completed games, not decisions or 40-second rollout segments. Update IDs are deduplicated when replayed. Supply `--elapsed-seconds` to use a measured wall-clock duration; otherwise the report uses upstream segment durations and excludes downtime. The supervisor's status uses wall time for the current invocation.

## Stop and recover

Create `runs/<name>/STOP` to stop the trainer. It polls every ten seconds. The supervisor also stops at the time/game budget, low disk space, rejected native actions, or a 30-minute stall. Stopping can lose work since the last strict checkpoint. It terminates only the trainer process group it started; emulator workers are managed separately.

On trainer failure it attempts at most two restarts from an existing strict checkpoint, retaining configuration. A nonempty output directory is never reused. Engine failures may still require manual recovery; there is no claim that all crashes recover automatically. Reboot/autostart and sleep settings are not changed.

The upstream `latest.pt` points to the latest strict resumable checkpoint, not the strongest model. Keep evaluated winners separately; the wrapper does not automatically certify or promote a best model. Avoid untrusted checkpoints, which may be deserialized by upstream PyTorch loading code.
