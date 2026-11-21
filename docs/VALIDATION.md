# Validation status

Checked November 20, 2026. Real simulator training and week-long throughput have not been tested because the compatible engine and Linux training topology are absent.

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
