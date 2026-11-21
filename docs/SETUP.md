# Setup and prerequisites

## Desktop

Use Python 3.12 and the isolated `.venv`. The bootstrap script installs PyTorch 2.7.1 from the official CUDA 12.8 index because upstream's older 2.5.1 pin predates RTX 50-series support. This is an integration change; test the loaded model and relevant upstream tests before training. Do not install the upstream requirements afterward, as that would downgrade PyTorch.

## Offline game environment

Follow the pinned [FirstLight installation guide](../vendor/FirstLight_CR/docs/getting_started.md). It requires:

- A dedicated Android 12 MuMu instance with root and ARM64 support.
- A locally supplied Null's Royale **15.535.13 / ARM64** APK and matching **15.535.86** downloaded content.
- JDK 17, Android NDK, and Android SDK build-tools if building the offline APK.

Use the upstream checks before building or replacing an emulator installation. The exact package/content identities are recorded in `vendor/FirstLight_CR/native_runner/supported_engine.json`. Do not substitute the latest APK or bypass compatibility checks. Do not overwrite an instance holding personal game data.

This is the blocker for real match evaluation. State of this PC on November 20, 2026:

- MuMu Player 6.8.16 is installed. The dedicated engine instance is index 1, `Android Device-1`: Android 12, ARM64 support through MuMu's translator, root on (`adb root` gives `uid=0`), 4 vCPUs, ADB at `127.0.0.1:16416`, nothing installed. (`MuMuManager main engine install` cannot fetch the Android 12 image by itself; the instance was created from the MuMu window.)
- Installed for the build, outside the repository: Microsoft OpenJDK 17.0.21 (`%LOCALAPPDATA%\Android\jdk-17.0.21.1+2`, checksum-verified), and under `%LOCALAPPDATA%\Android\Sdk` the command-line tools 23.0, platform-tools 37.0.3, build-tools 37.0.0 and NDK 27.3.13750724.
- `vendor/FirstLight_CR/.env` (git-ignored) points at those tools, MuMu's manager and instance 1; both of upstream's settings loaders read it.
- The project lives at `%USERPROFILE%\AI-Projects\Machine Learning\ClashAi`. The venv moved with it; its `python.exe` works, but launchers such as `pip.exe` still name the old folder, so `python -m pip` is the way to call pip.
- Still missing: the APK below.

Real matches need, in order:

1. MuMu Player 12 with an Android 12 instance that has root and ARM64 support enabled.
2. The Null's Royale APK 15.535.13, arm64-v8a, SHA-256 `685c115ee432d5fd2d7be856a58ea0cb029c9c187aa9061f3cd250994dfbcf2a`, at `vendor/FirstLight_CR/user-provided/nulls-royale.apk`. It is user-supplied; nothing in this repository downloads it.
3. Its in-game content update 15.535.86, fetched by the game itself once; `check_original_game.ps1` then cuts the instance off from the network. If the update server now delivers a different content version, the fingerprint check fails and this engine pin cannot be built.
4. JDK 17, Android SDK platform-tools (ADB) and build-tools, and the Android NDK (upstream used NDK 27.3 and build-tools 37.0) to build the offline APK.
5. `vendor/FirstLight_CR/.env` filled in as the upstream guide describes, then `check_original_game.ps1`, `build_offline_engine.ps1`, `install_offline_engine.ps1` and `native_runner/start_offline.ps1`.

With the engine listening, `scripts/benchmark.py run --only giant-graveyard --games 2` is the first real match check: its report lists whether an evolution and an ability fired.

## Generalist deck pool

For a quick environment smoke test, run `scripts/build_starter_pool.py`. It validates the 20 upstream example decks against supported card/form masks and creates `local/starter-deck-pool.json` and `local/example-match.json`. These are examples, not a complete meta dataset. The generalist configuration deliberately points to the separate broad `local/deck-pool.json` so the small starter pool is not mistaken for full training coverage.

Download the public [IL_Replay dataset](https://huggingface.co/datasets/VanguardX101/IL_Replay) when storage and the engine are ready. Place it under `local/datasets/IL_Replay/replays/`. Build the pool using the upstream parser, which retains exact evolution/hero forms and tower troops:

```powershell
$env:PYTHONPATH = (Resolve-Path vendor/FirstLight_CR).Path
.venv/Scripts/python.exe -m native_runner.training.v4.build_ppo_deck_pool --dataset-root local/datasets --output local/deck-pool.json --minimum-uses 5 --focus-decks vendor/FirstLight_CR/native_runner/training/v4/focus_decks_20.json --expected-pool-size 7089
```

From the public data this gives 7,089 decks, all 20 focus decks included (built and hash-checked on November 20, 2026). The upstream trainer, however, refuses any pool other than 7,695 decks with 20 focus decks (`train_ppo_self_play_cluster.py:1171`), and no usage threshold on the public data produces 7,695: four uses gives 8,530 and five gives 7,089. That figure belongs to the author's own pool, which is not published. `scripts/local_ppo.py` therefore loads the trainer with that one condition reduced to the 20-focus-deck part, leaving the file on disk untouched; it refuses to start if the line ever moves, and `scripts/train.py` applies the same rule before any engine starts. Held-out evaluation decks come from replay decks used one to four times, which the five-use threshold keeps out of training; `scripts/benchmark.py build-suite` selects them, including a held-out Giant Graveyard list (see the runbook). The `replays/` part of the dataset is about 0.83 GB in 52 Parquet files; the `actions/` part (1.06 GB) is only needed for imitation learning. The pretrained General weights load without this dataset.

## Training host

Training runs on this Windows PC with one GPU, no Linux, WSL or Slurm. Upstream's trainer, engine-worker service and engines already talk over local TCP; what tied them to Linux was a handful of system calls. `scripts/local_ppo.py` supplies Windows equivalents when it is imported, without editing the pinned upstream, and `scripts/train.py` uses it on both platforms. No cloud resource is purchased or rented by these scripts.

### What the trainer depends on, and what replaces it on Windows

Audited at the pinned commit; line numbers refer to `native_runner/training/v4/`.

| Dependency | Source | On Windows |
| --- | --- | --- |
| NCCL process group | `train_ppo_self_play_cluster.py:1117` | gloo, which handles the trainer's CUDA-tensor collectives (checked on this GPU); one rank needs no `torchrun` |
| CPU pinning via `os.sched_getaffinity` / `sched_setaffinity` | trainer `:564`, `:645`, `:1152`; `async_cluster_self_play.py:872`, `:1727`; worker service `:106` | All CPUs reported, pinning skipped; Windows schedules the threads |
| `fork` start method for engine workers | `remote_ppo_worker_server.py:123` | `spawn`; the accepted socket passes to the worker through `multiprocessing` |
| Closing the inherited listener in each worker | `remote_ppo_worker_server.py:48` | Nothing is inherited under `spawn`, so nothing is closed |
| `prctl(PR_SET_PDEATHSIG)` through `ctypes.CDLL(None)` | `async_cluster_self_play.py:856` | Skipped; workers end when the service stops them or their trainer connection closes, and the supervisor ends the whole process tree |
| Android Emulator guests with `/dev/kvm`, `taskset`, `qemu-img` | `launch_ppo_engine_node.sh` | Upstream's own Windows engine cluster, `native_runner/start_engine_cluster.ps1`: one MuMu VM with 1–24 engines of 8 resident slots |
| `SLURM_JOB_ID` equal to the topology's job ID | trainer `:857` | A single-PC topology carries the job ID `local`; the trainer's host check still ties it to this machine |
| Exactly 7,695 decks and 20 focus decks | trainer `:1171` | Reduced to the 20-focus check on every platform, as above |

### Running it on this PC

1. The offline engine from the section above, with `CR_SLOTS_PER_ENGINE=8` and `CR_BUILD_ENGINE_COUNT` / `CR_ENGINE_COUNT` set to the number of engines (upstream training guide, section 5). The PPO trainer requires exactly eight resident slots per engine.
2. `vendor/FirstLight_CR/native_runner/start_engine_cluster.ps1` to start them on `127.0.0.1:26789` and up.
3. A topology from the engines that answer. It queries each one's status and attestation and applies the acceptance checks of upstream's Linux node launcher, minus Slurm:

```powershell
.venv/Scripts/python.exe scripts/local_ppo.py topology --engines 6
```

4. A pilot: `scripts/train.py --config configs/giant-graveyard.json --hours 2 --run-name gy-pilot`. For a single-PC topology the supervisor starts upstream's engine-worker service itself, logs it to `runs/<run>/worker-service.log`, and stops it with the trainer.

Evaluation and training share the engine ports, so the cluster and a `benchmark.py` run do not run at the same time, as upstream also requires.

The Linux route (native Ubuntu with single-node Slurm and Android Emulator guests, upstream training guide sections 6 and 7) still works through the same `scripts/train.py`; on Linux it keeps NCCL and CPU pinning.

### Throughput arithmetic

100,000 games in 168 hours is 595 completed games per hour. Upstream trained General with 1,528 concurrent matches and eight GPUs. The Windows cluster runs one engine per guest vCPU with eight resident matches each; on this PC's 16 logical CPUs, six to twelve engines leaves room for the trainer, or 48 to 96 concurrent matches. At 48 concurrent matches, each match must finish within about 4.8 minutes of wall time, roughly real time for a full-length battle, while eight CPU cores run all of them and the GPU also performs PPO updates. Whether that holds is unknown until a pilot measures it; `scripts/lab.py report` reports measured completed games per hour.

## Pilot, then unattended training

With the engines running and `local/topology.json` built:

```powershell
.venv/Scripts/python.exe scripts/train.py --hours 2 --run-name pilot
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

The Giant Graveyard specialist uses the same gates with its own recipe: `scripts/train.py --config configs/giant-graveyard.json --hours 2 --run-name gy-pilot`. It follows upstream's Hog 2.6 specialist recipe (start from General, fixed learner deck, no self-play, 60% frozen opponents and 40% IL anchor, opponent decks 30% focus, 40% usage-weighted and 30% hard) and refuses to start if the deck is missing from the pool.
