"""Inspect the pinned generalist and run real offline evaluation; no account automation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "vendor" / "FirstLight_CR"
sys.path.insert(0, str(UPSTREAM))


def config():
    return json.loads((ROOT / "configs/generalist.json").read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def coverage():
    source = UPSTREAM / "native_runner/data/competitive/card_support.json"
    cards = json.loads(source.read_text(encoding="utf-8"))["supported_cards"]
    result = {
        "source_commit": config()["upstream_commit"],
        "scope": "Pinned engine catalog, not a claim of current live-game parity",
        "cards": len(cards),
        "evolution_capable_cards": sum(bool(c["form_availability"] & 1) for c in cards),
        "hero_or_champion_capable_cards": sum(bool(c["form_availability"] & 2) for c in cards),
        "catalog": cards,
    }
    save_json(ROOT / "local/coverage.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "catalog"}, indent=2))
    return 0


def verify_model():
    import torch
    from native_runner.training.v4.policy_session import load_policy_v4

    path = ROOT / config()["checkpoint"]
    manifest = json.loads((UPSTREAM / "checkpoints/manifest.json").read_text())
    expected = next(row for row in manifest["models"] if row["path"] == path.relative_to(UPSTREAM / "checkpoints").as_posix())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected["sha256"]:
        raise ValueError("Pretrained checkpoint hash does not match upstream release manifest")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; install the CUDA 12.8 PyTorch wheel")
    torch.set_num_threads(2)
    loaded = load_policy_v4(path, device="cuda")
    # Exercise the GPU kernels independently of the absent game engine.
    x = torch.randn(512, 512, device="cuda", requires_grad=True)
    (x @ x.T).square().mean().backward()
    torch.cuda.synchronize()
    if not torch.isfinite(x.grad).all():
        raise RuntimeError("CUDA backward pass produced nonfinite values")
    result = {
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "parameters": sum(p.numel() for p in loaded.model.parameters()),
        "checkpoint_sha256": digest,
        "checkpoint_id": loaded.checkpoint_id,
        "model_loaded_on_cuda": True, "cuda_forward_backward": "passed",
        "gameplay_tested": False,
    }
    save_json(ROOT / "local/model-verification.json", result)
    print(json.dumps(result, indent=2))
    return 0


def doctor():
    checks = []
    def add(name, passed, detail):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})
    cfg = config()
    add("python", sys.version_info[:2] == (3, 12), sys.version.split()[0])
    commit = subprocess.run(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], capture_output=True, text=True)
    add("pinned_upstream", commit.returncode == 0 and commit.stdout.strip() == cfg["upstream_commit"], commit.stdout.strip())
    add("general_checkpoint", (ROOT / cfg["checkpoint"]).is_file(), cfg["checkpoint"])
    try:
        import torch
        add("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unavailable")
        add("training_backend", sys.platform == "linux" and torch.distributed.is_nccl_available(), "Upstream PPO requires Linux/CUDA/NCCL and a prepared engine topology")
    except ImportError as error:
        add("torch", False, str(error))
    for field in ("topology", "deck_pool"):
        add(field, (ROOT / cfg[field]).is_file(), cfg[field])
    engine = json.loads((UPSTREAM / "native_runner/supported_engine.json").read_text())
    add("engine_readiness", (ROOT / cfg["topology"]).is_file(), f"User-supplied offline engine: APK {engine['apk_version']}, content {engine['runtime_content_version']}; a file alone is not runtime verification")
    result = {"platform": platform.platform(), "checks": checks, "training_ready": all(c["passed"] for c in checks), "note": "Runtime connectivity and native parity still require a real smoke run."}
    save_json(ROOT / "local/doctor.json", result)
    print(json.dumps(result, indent=2))
    return 0 if result["training_ready"] else 2


def read_metrics(path):
    """Deduplicate replayed update IDs after resume; ignore an incomplete final line."""
    by_update = {}
    if not Path(path).exists():
        return []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            raise
        if row.get("stage") == "ppo":
            by_update[int(row["update"])] = row
    return [by_update[key] for key in sorted(by_update)]


def summarize_metrics(rows, elapsed_seconds=None):
    games = sum(int(r.get("completed_games_this_segment", 0)) for r in rows)
    seconds = elapsed_seconds if elapsed_seconds is not None else sum(float(r.get("end_to_end_seconds", 0)) for r in rows)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Elapsed time must be finite and nonnegative")
    rate = games * 3600 / seconds if seconds > 0 else 0
    return {
        "updates": len(rows), "completed_games": games, "elapsed_seconds": seconds,
        "games_per_hour": rate, "projected_games_seven_days": int(rate * 168),
        "required_games_per_hour_for_100k": 100000 / 168,
        "native_rejected_actions": sum(int(r.get("native_rejected_actions", 0)) for r in rows),
        "projection_is_not_a_strength_measure": True,
    }


def evaluate(args):
    cfg = config()
    match = Path(args.match).resolve()
    if not match.is_file():
        raise ValueError("Supply a real MatchConfig JSON; no default Hog mirror is substituted")
    out = ROOT / "runs" / time.strftime("evaluation-%Y%m%d-%H%M%S")
    env = dict(os.environ, PYTHONPATH=str(UPSTREAM), WANDB_MODE="offline")
    command = [sys.executable, "-m", "native_runner.training.v4.evaluate", "--checkpoint-a", str(Path(args.candidate).resolve() if args.candidate else ROOT / cfg["checkpoint"]), "--checkpoint-b", str(Path(args.opponent).resolve() if args.opponent else ROOT / cfg["opponents"][2]), "--games", str(args.games), "--match-config", str(match), "--output", str(out), "--host", args.host, "--port", str(args.port), "--device", "cuda", "--seed", "20261120"]
    return subprocess.call(command, cwd=UPSTREAM, env=env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("doctor", "coverage", "verify-model"):
        sub.add_parser(name)
    report = sub.add_parser("report")
    report.add_argument("run", type=Path)
    report.add_argument("--elapsed-seconds", type=float)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--match", required=True)
    ev.add_argument("--games", type=int, default=200)
    ev.add_argument("--candidate")
    ev.add_argument("--opponent")
    ev.add_argument("--host", default="127.0.0.1")
    ev.add_argument("--port", type=int, default=26789)
    args = parser.parse_args()
    if args.command == "report":
        print(json.dumps(summarize_metrics(read_metrics(args.run / "metrics.jsonl"), args.elapsed_seconds), indent=2))
        return 0
    if args.command == "evaluate":
        if args.games < 1:
            parser.error("games must be positive")
        return evaluate(args)
    return {"doctor": doctor, "coverage": coverage, "verify-model": verify_model}[args.command]()


if __name__ == "__main__":
    raise SystemExit(main())
