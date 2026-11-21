# Setup and prerequisites

## Desktop

Use Python 3.12 and the isolated `.venv`. The bootstrap script installs PyTorch 2.7.1 from the official CUDA 12.8 index because upstream's older 2.5.1 pin predates RTX 50-series support. This is an integration change; test the loaded model and relevant upstream tests before training. Do not install the upstream requirements afterward, as that would downgrade PyTorch.

## Offline game environment

Follow the pinned [FirstLight installation guide](../vendor/FirstLight_CR/docs/getting_started.md). It requires:

- A dedicated Android 12 MuMu instance with root and ARM64 support.
- A locally supplied Null's Royale **15.535.13 / ARM64** APK and matching **15.535.86** downloaded content.
- JDK 17, Android NDK, and Android SDK build-tools if building the offline APK.

Use the upstream checks before building or replacing an emulator installation. The exact package/content identities are recorded in `vendor/FirstLight_CR/native_runner/supported_engine.json`. Do not substitute the latest APK or bypass compatibility checks. Do not overwrite an instance holding personal game data.

The current setup does not contain these game files or a prepared MuMu instance. This is the blocker for real match evaluation.

## Generalist deck pool

For a quick environment smoke test, run `scripts/build_starter_pool.py`. It validates the 20 upstream example decks against supported card/form masks and creates `local/starter-deck-pool.json` and `local/example-match.json`. These are examples, not a complete meta dataset. The generalist configuration deliberately points to the separate broad `local/deck-pool.json` so the small starter pool is not mistaken for full training coverage.

Download the public [IL_Replay dataset](https://huggingface.co/datasets/VanguardX101/IL_Replay) when storage and the engine are ready. Place it under `local/datasets/IL_Replay/replays/`. Build the pool using the upstream parser, which retains exact evolution/hero forms and tower troops:

```powershell
$env:PYTHONPATH = (Resolve-Path vendor/FirstLight_CR).Path
.venv/Scripts/python.exe -m native_runner.training.v4.build_ppo_deck_pool --dataset-root local/datasets --output local/deck-pool.json --minimum-uses 5 --focus-decks vendor/FirstLight_CR/native_runner/training/v4/focus_decks_20.json
```

The pool must have at least 20 decks for this launcher; use the full replay-derived pool for a serious generalist. Review card/form coverage and reserve unseen deck combinations for evaluation. The pretrained General weights can be loaded without downloading this dataset.

## Training host

The upstream production trainer is not a Windows or simple WSL command. It depends on Linux, CUDA/NCCL, CPU affinity, remote collection workers, and a real Slurm allocation described by an engine readiness topology. Follow [upstream training documentation](../vendor/FirstLight_CR/docs/training.md) to provision it. No cloud resource is purchased or rented by these scripts.

On that host, create a separate Python 3.12 environment; a Windows venv cannot be reused:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements.txt
```

Put the verified topology in `local/topology.json` and the validated pool in `local/deck-pool.json`. This repository's launcher deliberately refuses Windows and an absent/mismatched topology. It does not fake Slurm variables to get past upstream validation. Adapting the trainer to a stand-alone desktop remains additional work; it has not been implemented or benchmarked here.

## Pilot, then unattended training

Within the prepared training allocation:

```bash
.venv/bin/python scripts/train.py --hours 2 --run-name pilot
```

Review `runs/pilot/status.json`, per-attempt metrics and logs, and real replays. Check that completed games are increasing, the optimizer is updating, actions are not rejected, and output checkpoints are loadable. Test an explicit recovery with `--resume /absolute/path/to/latest.pt` in a new short run using the same pool, opponents, topology and hyperparameters. This may require resetting failed simulation workers before recovery.

Longer than four hours requires a reviewed `local/pilot-approved.json`:

```json
{
  "upstream_commit": "4c82496ad5af49a3c7ea2392f26ff7bac33d9707",
  "engine_verified": true,
  "resume_verified": true
}
```

Write this file only after the corresponding real checks pass. It is a manual evidence gate, not an automatic certification. Then run `scripts/train.py --hours 168 --run-name week-1`. At least 595 completed games/hour is required for 100,000 games in a week. This hardware has not yet been measured against that target.
