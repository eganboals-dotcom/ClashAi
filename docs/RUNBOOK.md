# Runbook

## Check the setup

Use the project's Python executable with `scripts/lab.py doctor`, `coverage`, and `verify-model`. Reports go to ignored `local/` files. Hash verification and GPU matrix operations are not game-playing benchmarks.

## Evaluate strength

`scripts/benchmark.py` plays a deck suite against frozen references through upstream's own `evaluate()`, so game handling, decision timing and owner swapping are upstream's. It needs the offline engine listening locally (default `127.0.0.1:26789`) and exits with code 2 when none answers.

```powershell
.venv/Scripts/python.exe scripts/benchmark.py validate
.venv/Scripts/python.exe scripts/benchmark.py run --only giant-graveyard --games 2 --name smoke
.venv/Scripts/python.exe scripts/benchmark.py run --promote --name general-baseline
.venv/Scripts/python.exe scripts/benchmark.py run --candidate runs/<run>/attempt-00/latest.pt --promote
```

`configs/benchmark-suite.json` holds 11 decks and 12 matchups across four tower troops. Two Giant Graveyard lists (Hero Giant, Evo Valkyrie, Evo Bomber, with Princess and Cannoneer towers) are composed for coverage rather than taken from replays; the other nine are upstream focus decks and are therefore part of training.

- **Protocol.** Each matchup runs in both deck orientations, so the candidate plays each deck. Within an orientation upstream alternates engine owners. Both sides use the same five-tick decision interval, headless, with argmax decoding and a fixed seed. The default references are General, Active IL and Hog 2.6 specialist 2, the trainer's frozen opponents. The full suite at the default 10 games per orientation is 720 games, played one at a time on one engine.
- **Report.** `runs/benchmarks/<name>/report.md` and `report.json` give, per reference and per matchup: score (wins plus half of draws), Wilson 95% interval, W-L-D, truncations, rejected actions, evolutions played and abilities activated by each side, wins over a passive reference (three or fewer plays, the exploit described in upstream's training history), wall time and games per hour. Matchups whose interval lies wholly below 0.5 are listed. The overall score averages per-matchup scores, so each matchup counts equally and one favorable deck cannot hide a loss elsewhere. `matchups_with_evolution_and_ability` lists matchups in which both an evolution and an ability actually fired. Per-game rows are in each `games.jsonl`.
- **Precision.** Twenty games per matchup and reference give intervals of about ±0.2; the 720-game overall score is good to roughly ±0.04. Small overall differences are noise.
- **Keeping the best.** With `--promote`, a candidate replaces the checkpoint in `runs/best/<track>/` (the suite file name, plus the candidate deck when one is fixed) only if it had no rejected actions, was measured on the same suite, references, game count, seed and upstream commit, scores higher overall, and its worst matchup is no more than 0.05 below the incumbent's. The first promoted run, General itself, sets the baseline. A newer or longer-trained checkpoint is never promoted on that basis alone.
- **Held-out decks.** `--pool local/deck-pool.json` marks each deck as in or out of training; a matchup counts as held out when either deck is outside the pool. With the replay pool present, `build-suite` adds the most-used Giant Graveyard deck in the pool, the most-used one outside it, and six distinct held-out decks with their matchups:

```powershell
$env:PYTHONPATH = (Resolve-Path vendor/FirstLight_CR).Path
.venv/Scripts/python.exe -m native_runner.training.v4.build_ppo_deck_pool --dataset-root local/datasets --output local/deck-pool-all.json --minimum-uses 1
.venv/Scripts/python.exe scripts/benchmark.py build-suite --pool local/deck-pool.json --all-decks local/deck-pool-all.json
.venv/Scripts/python.exe scripts/benchmark.py run --suite local/benchmark-suite.json --pool local/deck-pool.json --promote
```

### Giant Graveyard specialist

`configs/giant-graveyard.json` fixes the learner deck to Evo Witch, Hero Bowler, Evo Giant Snowball, Graveyard, Giant, Guards, Zappies and Arrows with Tower Princess (94 replay uses, inside the training pool). `specialist-suite` pits that deck against the 20 focus decks, the ten most-used distinct pool decks, ten held-out decks, the most-used other Giant Graveyard deck in the pool and a held-out Giant Graveyard deck. `--candidate-deck` keeps only the orientation in which the candidate plays that deck, since a specialist is judged on its own deck; the reference still plays every opponent deck. The full suite is 42 matchups, 1,260 games at the default 10 per matchup and reference, and its best checkpoint is kept apart from the generalist's, in `runs/best/giant-graveyard-suite-giant-graveyard/`:

```powershell
.venv/Scripts/python.exe scripts/benchmark.py specialist-suite --pool local/deck-pool.json --all-decks local/deck-pool-all.json
.venv/Scripts/python.exe scripts/benchmark.py run --suite local/giant-graveyard-suite.json --pool local/deck-pool.json --candidate-deck giant-graveyard --promote --name gy-general-baseline
.venv/Scripts/python.exe scripts/benchmark.py run --suite local/giant-graveyard-suite.json --pool local/deck-pool.json --candidate-deck giant-graveyard --candidate runs/<run>/attempt-00/latest.pt --promote
```

The first run scores General with this deck and sets the baseline that a trained specialist has to beat.

`scripts/lab.py evaluate --match <MatchConfig.json>` still runs a single matchup through the upstream command line.

## Monitor training

Read `runs/<name>/status.json` and `console-*.log`. Each attempt has its own upstream metrics, matches, league snapshots, and strict resume checkpoint. Report one attempt with:

```powershell
.venv/Scripts/python.exe scripts/lab.py report runs/week-1/attempt-00
```

The report counts completed games, not decisions or 40-second rollout segments. Update IDs are deduplicated when replayed. Supply `--elapsed-seconds` to use a measured wall-clock duration; otherwise the report uses upstream segment durations and excludes downtime. The supervisor's status uses wall time for the current invocation.

## Stop and recover

Create `runs/<name>/STOP` to stop the trainer. It polls every ten seconds. The supervisor also stops at the time/game budget, low disk space, rejected native actions, or a 30-minute stall. Stopping can lose work since the last strict checkpoint. It terminates only the trainer process group it started; emulator workers are managed separately.

On trainer failure it attempts at most two restarts from an existing strict checkpoint, retaining configuration. A nonempty output directory is never reused. Engine failures may still require manual recovery; there is no claim that all crashes recover automatically. Reboot/autostart and sleep settings are not changed.

The upstream `latest.pt` points to the latest strict resumable checkpoint, not the strongest model. Training never promotes a checkpoint; only `benchmark.py run --promote` does, under the rule above, and `runs/best/<track>/best.json` records which file won and on which report. Avoid untrusted checkpoints, which may be deserialized by upstream PyTorch loading code.
