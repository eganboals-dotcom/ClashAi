# ClashAi

A general-deck Clash Royale AI research workspace, built on a pinned [FirstLight CR](https://github.com/Jaasssoooonnnnn/FirstLight_CR) revision. Start from the pretrained **General** policy, then evaluate and improve it in a compatible offline simulator.

**Status: pretrained General loaded on an RTX 5070 Ti; CUDA checks and 30 tests passed. Training has not started.** The compatible game engine and Linux training topology are separate prerequisites and are not bundled. Supporting a card in the versioned catalog does not prove parity with the current live game.

## What is included

- Pinned upstream source, pretrained models, and engine compatibility tables via a Git submodule.
- A generalist configuration: no fixed learner deck, 50% self-play and 50% frozen opponents, including General, Active IL, and the Hog specialist.
- RTX 50-series-compatible CUDA 12.8 PyTorch installation.
- Checkpoint hash verification, CUDA checks, and a card/form coverage report.
- Actual offline model-versus-model evaluation through FirstLight's native evaluator.
- A bounded Linux PPO supervisor with logging, strict-checkpoint recovery, time/game limits, disk and stall checks, and a stop file.
- Tests for metrics accounting, projections, generalist arguments, and resume arguments.

This is an offline research and practice project. It contains no official-account login, ranked queue automation, or anti-cheat evasion. A game-count target is not a strength guarantee.

## Start on Windows

```powershell
git clone --recurse-submodules https://github.com/eganboals-dotcom/ClashAi.git
cd ClashAi
./scripts/bootstrap.ps1 -Python 'C:\path\to\Python312\python.exe'
.venv/Scripts/python.exe scripts/lab.py doctor
.venv/Scripts/python.exe scripts/lab.py coverage
.venv/Scripts/python.exe scripts/lab.py verify-model
.venv/Scripts/python.exe scripts/build_starter_pool.py
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

If cloned without submodules, run `git submodule update --init --recursive` first. The parent repository stays small; the upstream includes approximately 240 MB of pretrained checkpoints.

`doctor` intentionally exits with code 2 while training prerequisites are missing. Windows can load models, run inference, and evaluate against a prepared offline MuMu engine. The pinned upstream PPO backend requires a **Linux CUDA/NCCL training host and prepared Slurm engine topology**. Installing MuMu or WSL alone does not satisfy that requirement.

See [setup and remaining prerequisites](docs/SETUP.md), [runbook](docs/RUNBOOK.md), and [validation status](docs/VALIDATION.md). Local paths, telemetry, datasets, game files, and generated checkpoints are ignored by Git.

## Scope of “any deck”

The model accepts arbitrary legal decks within the pinned engine's catalog: **122 cards, 42 evolution-capable cards, and 24 hero/champion-capable cards** (categories overlap). Training is configured for a broad deck pool rather than a Giant Graveyard specialization. It does **not** mean every possible deck has been trained or that all current cards exist in this older engine. Run the coverage command for exact supported IDs and names.

Upstream revision: `4c82496ad5af49a3c7ea2392f26ff7bac33d9707`.

## Attribution

FirstLight CR and its contributors provide the policy, training algorithms, native environment, models, and versioned compatibility data under Apache-2.0. Their license and notice remain in the submodule. The wrapper scripts in this repository are also Apache-2.0. No rights to third-party game binaries or assets are granted by this repository.
