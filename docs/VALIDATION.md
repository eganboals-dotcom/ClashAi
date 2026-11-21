# Validation status

Checked November 20, 2026. Real simulator training and week-long throughput have not been tested because the compatible engine is absent. Earlier passes below describe the trainer as Linux-only; the Windows section at the end supersedes that.

Passed:

- Python 3.12.10 isolated environment and dependencies installed.
- PyTorch 2.7.1+cu128 recognizes the RTX 5070 Ti.
- General checkpoint SHA-256 matches upstream manifest: `2fb7a94fa5df0f12bd72433874487bc7d84a4efb2a6dac827d7adff0a5eef083`.
- General policy (12,053,037 parameters) loads onto CUDA.
- Independent CUDA matrix forward/backward operations produce finite gradients. This is a GPU installation check, not policy training.
- 5 wrapper unit tests passed.
- 25 upstream model and PPO padding tests passed on this environment.
- Coverage report generated from the pinned catalog: 122 cards, 42 evolution-capable and 24 hero/champion-capable cards.
- Windows launch correctly refuses the unsupported NCCL trainer instead of claiming to start training.

Not performed: real matches, PPO optimization on real match rollouts, recovery from a real training failure, coverage of every current live-game card, 100,000-game throughput, or any ranked performance test.

## Second pass, November 20, 2026

Engine search: no MuMu installation, APK, ADB, JDK, Android SDK/NDK or WSL distribution exists on this PC, so no match, training step or throughput measurement was possible. `docs/SETUP.md` lists the exact missing items.

Passed:

- `configs/benchmark-suite.json` resolves through upstream `deck_identity_from_keys` and the pinned card/form catalog: 11 decks, 12 matchups, four tower troops. Both Giant Graveyard lists resolve with two evolution slots and one hero slot. Against the 20-deck starter pool, the composed Giant Graveyard decks are correctly reported as outside it and the focus decks inside.
- `benchmark.py run --dry-run` planned 72 runs (12 matchups × 2 orientations × 3 references), 720 games.
- A non-dry run loaded the candidate and reference checkpoints on CUDA, constructed upstream's battle environment, failed at the engine's attestation request because nothing listens on port 26789, and exited with code 2 and a plain message.
- 13 wrapper unit tests pass, 8 of them new for suite validation, held-out selection, per-matchup scoring, Wilson intervals, the keep-best rule and owner-to-policy event mapping.
- Trainer dependencies audited (table in `docs/SETUP.md`): Slurm is only a job-ID identity check; NCCL, CPU pinning, `fork` and Android Emulator guests are Linux-only; the trainer accepts only the 7,695/20 pool. `scripts/train.py` now enforces that pool size before starting.

Replay data and deck pools, same day:

- The `replays/` part of IL_Replay (52 Parquet files, 788 MB, 252,238 anonymized replays, 504,476 player-deck uses) downloaded to `local/datasets/IL_Replay/`; every file matches the dataset's `SHA256SUMS`.
- Upstream's pool builder found 53,202 distinct decks with tower troops and no invalid decks. At five or more uses the pool has 7,089 decks including all 20 focus decks; it is saved as `local/deck-pool.json`. No threshold yields upstream's required 7,695 (four uses: 8,530), so the trainer's pool check cannot be met from public data.
- `benchmark.py build-suite` produced `local/benchmark-suite.json`: 19 decks, 27 matchups, 1,620 games at the default 10 per orientation. It adds the most-used Giant Graveyard deck in training (1,561 uses: Giant, Graveyard, Bowler, Hero Berserker, Evo Giant Snowball, Evo Witch, Zappies, Arrows), a held-out Giant Graveyard deck (4 uses: Giant, Graveyard, Hero Bowler, Evo Archers, Evo Giant Snowball, Guards, Zappies, Arrows) and six distinct held-out decks. The pool holds 58 Giant Graveyard decks; 306 more appear only below the five-use threshold.
- Validated against the pool: the replay Giant Graveyard deck and the nine focus decks are inside it; both composed Giant Graveyard lists, the held-out Giant Graveyard deck and all six held-out decks are outside it. All replay-derived held-out decks use the Princess tower; tower-troop variety comes from the starter decks.

Giant Graveyard specialist, same day:

- `configs/giant-graveyard.json` resolves to pool deck `78319a36…7d44` with 94 uses: Evo Witch, Hero Bowler, Evo Giant Snowball, Graveyard, Giant, Guards, Zappies, Arrows, Tower Princess. The fourth-row card was read from a screenshot as Zappies; a Tesla version of the list is not in the pool.
- Upstream's own argument parser accepted the full specialist command (`--learner-deck-id`, `--fixed-learner-opponent-mix`, the 30/40/30 opponent mix, the IL anchor); the run stopped at the distributed rendezvous, as expected without `torchrun` on Linux, before creating any output.
- `specialist-suite` produced `local/giant-graveyard-suite.json`: 43 decks, 42 matchups, 11 of them held out. A dry run planned 126 runs (42 matchups × 3 references, one orientation) and 1,260 games, with the candidate on Giant Graveyard in every run.
- 16 wrapper unit tests pass, including the specialist suite, `--candidate-deck` planning and the specialist trainer command.

Windows training path, same day:

- PyTorch 2.7.1+cu128's gloo backend on Windows ran the trainer's collectives on a CUDA tensor on the RTX 5070 Ti: `all_reduce` (sum and max), `all_gather_object`, `broadcast_object_list`, `barrier`.
- `scripts/train.py --config` (Giant Graveyard recipe) with a one-engine smoke topology and no engine, on this PC: the supervisor started upstream's engine-worker service, which loaded its catalogs in 2.9 s and listened; the trainer formed a gloo process group, loaded General, the IL anchor and the frozen opponents on CUDA, accepted the 7,089-deck pool and the topology, and the matchmaker built a match with the owner's Giant Graveyard deck (Evo Witch and Evo Giant Snowball as evolutions, Hero Bowler as hero). The trainer connected to the service, which spawned an engine worker through `spawn`; the worker asked for the match on `127.0.0.1:28000` and got "connection refused". The error travelled back to the trainer, the supervisor recorded `trainer_failed` after 46 s, and no Python process was left running.
- `local_ppo.py topology --engines 2` with no engine running exited with code 2 and wrote nothing.
- 22 wrapper unit tests pass; the 6 new ones cover the single-PC topology (upstream's own readers accept it; unready, mismatched or 4-slot engines are refused), the worker-service port mapping, the trainer's pool rule, the launcher command and the Windows replacements.

Not verified: a training update, a strict checkpoint and a resume. All three need rollouts, and rollouts need the engine.

Not measured: GPU learner throughput. Upstream's synthetic test batches contain a handful of units, so timing PPO on them would overstate real speed; a credible number needs real rollouts.
