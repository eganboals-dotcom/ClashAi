# ClashAi

A general-deck Clash Royale AI research workspace, built on a pinned [FirstLight CR](https://github.com/Jaasssoooonnnnn/FirstLight_CR) revision. Start from the pretrained **General** policy, then evaluate and improve it in a compatible offline simulator.

**Status: pretrained General loaded on an RTX 5070 Ti; CUDA checks, 25 upstream tests and 22 wrapper tests passed. Training now runs on Windows, without Linux or Slurm, through to the engine connection; no real match or training step has run because the offline engine is not installed on this PC.** The compatible game engine is a separate prerequisite and is not bundled. Supporting a card in the versioned catalog does not prove parity with the current live game.

## What is included

- Pinned upstream source, pretrained models, and engine compatibility tables via a Git submodule.
- A generalist configuration: no fixed learner deck, 50% self-play and 50% frozen opponents, including General, Active IL, and the Hog specialist.
- RTX 50-series-compatible CUDA 12.8 PyTorch installation.
- Checkpoint hash verification, CUDA checks, and a card/form coverage report.
- Actual offline model-versus-model evaluation through FirstLight's native evaluator.
- A per-matchup benchmark (`scripts/benchmark.py`) over a Giant Graveyard suite with both deck orientations, held-out labelling, evolution/ability evidence and a keep-best rule. It has not yet played a real match.
- A bounded single-GPU PPO supervisor for Windows or Linux (`scripts/train.py`, with `scripts/local_ppo.py` supplying Windows equivalents of upstream's Linux-only calls) with logging, strict-checkpoint recovery, time/game limits, disk and stall checks, and a stop file.
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

`doctor` intentionally exits with code 2 while training prerequisites are missing. Windows can load models, run inference, evaluate against a prepared offline MuMu engine, and train against upstream's Windows engine cluster; [setup](docs/SETUP.md#training-host) gives the steps. Installing MuMu alone does not satisfy that requirement: the offline engine has to be built and attested first.

See [setup and remaining prerequisites](docs/SETUP.md), [runbook](docs/RUNBOOK.md), and [validation status](docs/VALIDATION.md). Local paths, telemetry, datasets, game files, and generated checkpoints are ignored by Git.

## Scope of “any deck”

The model accepts arbitrary legal decks within the pinned engine's catalog: **122 cards, 42 evolution-capable cards, and 24 hero/champion-capable cards** (categories overlap). The generalist recipe trains on a broad deck pool; a second recipe, `configs/giant-graveyard.json`, trains a Giant Graveyard specialist on one fixed deck against that pool. It does **not** mean every possible deck has been trained or that all current cards exist in this older engine. Run the coverage command for exact supported IDs and names.

Upstream revision: `4c82496ad5af49a3c7ea2392f26ff7bac33d9707`.

## Attribution

FirstLight CR and its contributors provide the policy, training algorithms, native environment, models, and versioned compatibility data under Apache-2.0. Their license and notice remain in the submodule. The wrapper scripts in this repository are also Apache-2.0. No rights to third-party game binaries or assets are granted by this repository.
