"""Bounded Linux PPO supervisor for the pinned FirstLight engine topology.

This does not provision an engine, fabricate readiness, or automate a game account.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

from lab import ROOT, UPSTREAM, config, read_metrics, save_json, summarize_metrics


def command_for(cfg, output, updates, resume=None):
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=1", "-m", "native_runner.training.v4.train_ppo_self_play_cluster"]
    options = {
        "deck-pool": ROOT / cfg["deck_pool"], "checkpoint": ROOT / cfg["checkpoint"],
        "engine-ready": ROOT / cfg["topology"], "output-dir": output,
        "updates": updates, "self-play-percent": cfg["self_play_percent"],
        "fixed-opponent-percent": cfg["fixed_opponent_percent"],
        "learning-rate": cfg["learning_rate"], "lane-minibatch-size": cfg["lane_minibatch_size"],
        "lane-microbatch-size": cfg["lane_microbatch_size"], "sequence-chunk-steps": cfg["sequence_chunk_steps"],
        "rollout-segment-seconds": cfg["rollout_segment_seconds"], "ppo-epochs": cfg["ppo_epochs"],
        "seed": cfg["seed"], "snapshot-every": 100,
    }
    for key, value in options.items():
        command.extend(["--" + key, str(value)])
    for path in cfg["opponents"]:
        command.extend(["--fixed-opponent-checkpoint", str(ROOT / path)])
    if resume:
        command.extend(["--resume-checkpoint", str(resume)])
    return command


def stop_process(process):
    if process.poll() is not None:
        return
    # Only the process group created for this trainer is touched.
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    except ProcessLookupError:
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=2)
    parser.add_argument("--updates", type=int, default=100000)
    parser.add_argument("--run-name", default=time.strftime("generalist-%Y%m%d-%H%M%S"))
    parser.add_argument("--resume", type=Path, help="Explicit trusted strict PPO checkpoint")
    args = parser.parse_args()
    cfg = config()
    if not 0 < args.hours <= cfg["max_hours"] or args.updates < 1:
        parser.error("hours must be within (0, 168] and updates positive")
    if Path(args.run_name).name != args.run_name or args.run_name in {".", ".."}:
        parser.error("run-name must be a single directory name")
    if sys.platform != "linux":
        parser.error("The pinned PPO backend requires Linux/CUDA/NCCL. Windows supports model checks and offline evaluation only.")
    import torch
    if not torch.cuda.is_available() or not torch.distributed.is_nccl_available():
        parser.error("CUDA and NCCL are required")
    for key in ("checkpoint", "deck_pool", "topology"):
        if not (ROOT / cfg[key]).is_file():
            parser.error(f"Missing {key}: {cfg[key]}. See docs/SETUP.md")
    topology = json.loads((ROOT / cfg["topology"]).read_text())
    if topology.get("ok") is not True or topology.get("gpus") != 1:
        parser.error("A verified single-GPU engine topology is required")
    if str(topology.get("trainer_slurm_job_id")) != os.environ.get("SLURM_JOB_ID"):
        parser.error("Run within the Slurm allocation used to prepare this topology; do not invent a job ID")
    if str(topology.get("trainer_host", "")).split(".")[0] != socket.gethostname().split(".")[0]:
        parser.error("The topology belongs to a different trainer host")
    from native_runner.training.v4.deck_pool import DeckPoolManifestV4
    pool = DeckPoolManifestV4.load(ROOT / cfg["deck_pool"])
    if len(pool.decks) < 20:
        parser.error("Generalist training requires at least 20 validated decks; build a broad replay-derived pool")
    if args.hours > 4:
        pilot_path = ROOT / "local/pilot-approved.json"
        if not pilot_path.is_file():
            parser.error("Complete a 2–4 hour pilot first, then review it as described in docs/SETUP.md")
        pilot = json.loads(pilot_path.read_text())
        if pilot.get("upstream_commit") != cfg["upstream_commit"] or pilot.get("engine_verified") is not True or pilot.get("resume_verified") is not True:
            parser.error("Pilot evidence must match this revision and include engine and resume verification")
    run = ROOT / "runs" / args.run_name
    run.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    all_rows = {}
    resume = args.resume.resolve() if args.resume else None
    if resume and not resume.is_file():
        parser.error("Resume checkpoint does not exist")
    attempts = 0
    process = None
    reason = "not_started"
    env = dict(os.environ, PYTHONPATH=str(UPSTREAM), WANDB_MODE="offline", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1")
    try:
        while attempts <= cfg["max_restarts"]:
            attempt = run / f"attempt-{attempts:02d}"
            cmd = command_for(cfg, attempt, args.updates, resume)
            save_json(run / f"command-{attempts:02d}.json", cmd)
            last_progress = time.monotonic()
            last_update = -1
            with (run / f"console-{attempts:02d}.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(cmd, cwd=UPSTREAM, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                while process.poll() is None:
                    rows = read_metrics(attempt / "metrics.jsonl")
                    for row in rows:
                        all_rows[int(row["update"])] = row
                    if rows and int(rows[-1]["update"]) != last_update:
                        last_update = int(rows[-1]["update"])
                        last_progress = time.monotonic()
                    elapsed = time.monotonic() - started
                    summary = summarize_metrics(list(all_rows.values()), elapsed)
                    save_json(run / "status.json", {**summary, "state": "running", "attempt": attempts, "latest_checkpoint": str(attempt / "latest.pt") if (attempt / "latest.pt").exists() else None})
                    if summary["native_rejected_actions"]:
                        reason = "native_action_rejections"
                        break
                    if (run / "STOP").exists():
                        reason = "stop_file"
                        break
                    if elapsed >= args.hours * 3600:
                        reason = "time_budget"
                        break
                    if summary["completed_games"] >= cfg["target_games"]:
                        reason = "game_budget"
                        break
                    if shutil.disk_usage(run).free < cfg["minimum_free_gib"] * 1024**3:
                        reason = "low_disk_space"
                        break
                    if time.monotonic() - last_progress > cfg["stall_minutes"] * 60:
                        reason = "no_training_progress"
                        break
                    time.sleep(10)
                stop_process(process)
            for row in read_metrics(attempt / "metrics.jsonl"):
                all_rows[int(row["update"])] = row
            latest = attempt / "latest.pt"
            if latest.is_file():
                resume = latest.resolve()
            if reason != "not_started":
                break
            if process.returncode == 0:
                reason = "updates_complete"
                break
            if not resume or attempts >= cfg["max_restarts"]:
                reason = "trainer_failed"
                break
            # Upstream validates optimizer, RNG, topology, and hyperparameter identity.
            # Each attempt has a fresh output directory because upstream forbids reuse.
            attempts += 1
    except KeyboardInterrupt:
        reason = "interrupted"
    finally:
        if process is not None:
            stop_process(process)
        summary = summarize_metrics(list(all_rows.values()), time.monotonic() - started)
        save_json(run / "status.json", {**summary, "state": "stopped", "reason": reason, "latest_checkpoint": str(resume) if resume else None, "note": "Stopping can discard work after the last strict checkpoint; latest is not necessarily strongest."})
        print(json.dumps({"run": str(run), "reason": reason, **summary}, indent=2))
    return 0 if reason in {"updates_complete", "time_budget", "game_budget", "stop_file"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
