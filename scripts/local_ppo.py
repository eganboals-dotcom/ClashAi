"""Run upstream's PPO trainer and engine-worker service on one PC, Windows included.

The pinned upstream is not edited. On Windows the few Linux-only calls are given
local equivalents when this module is imported, and because it is also the main
module of every spawned worker process, those processes get them too.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from pathlib import Path
import signal
import socket
import sys
import types

from lab import LOCAL_JOB_ID, ROOT, UPSTREAM  # lab also puts the pinned upstream on sys.path

TRAINER = "native_runner.training.v4.train_ppo_self_play_cluster"
POOL_CHECK = "if len(manifest.decks) != 7_695 or len(manifest.focus_decks) != 20:"
WINDOWS = sys.platform == "win32"


class _NoParentDeathSignal:
    # shortcut: Windows has no PR_SET_PDEATHSIG; workers end when the service stops them or their trainer socket closes.
    def prctl(self, *_args):
        return 0


def _patch_windows():
    import ctypes

    import torch.distributed as dist
    from native_runner.training.v4 import async_cluster_self_play, remote_ppo_worker_server

    if not hasattr(os, "sched_getaffinity"):
        os.sched_getaffinity = lambda _pid: set(range(os.cpu_count()))
        # shortcut: no CPU pinning on Windows; the scheduler places threads. Pin with SetProcessAffinityMask if contention shows up.
        os.sched_setaffinity = lambda _pid, _cpus: None
    real_get_context = multiprocessing.get_context
    multiprocessing.get_context = lambda method=None: real_get_context("spawn" if method == "fork" else method)
    real_cdll = ctypes.CDLL
    async_cluster_self_play.ctypes = types.SimpleNamespace(
        CDLL=lambda name, *a, **k: _NoParentDeathSignal() if name is None else real_cdll(name, *a, **k))

    def prepare_spawned_worker(_listener_fd):
        # Under spawn the listener is not inherited, so there is no descriptor to close.
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)

    remote_ppo_worker_server._prepare_worker_process = prepare_spawned_worker
    real_init = dist.init_process_group
    dist.init_process_group = lambda backend=None, **k: real_init(backend="gloo" if backend == "nccl" else backend, **k)
    os.environ.setdefault("USE_LIBUV", "0")  # Windows PyTorch builds ship TCPStore without libuv


def load_trainer():
    """Import the upstream trainer with its pool-size rule relaxed to the public 20-focus pool."""
    path = UPSTREAM / "native_runner/training/v4/train_ppo_self_play_cluster.py"
    source = path.read_text(encoding="utf-8")
    if source.count(POOL_CHECK) != 1:
        raise RuntimeError("upstream pool check moved; re-check scripts/local_ppo.py against the pinned commit")
    module = types.ModuleType(TRAINER)
    module.__file__, module.__package__ = str(path), TRAINER.rpartition(".")[0]
    sys.modules[TRAINER] = module
    exec(compile(source.replace(POOL_CHECK, "if len(manifest.focus_decks) != 20:"), str(path), "exec"), module.__dict__)
    return module


def _argument(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


def run_trainer(argv):
    os.environ.update({key: value for key, value in
                       {"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29533", "RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0"}.items()
                       if key not in os.environ})
    topology = _argument(argv, "--engine-ready")
    if topology and json.loads(Path(topology).read_text(encoding="utf-8")).get("trainer_slurm_job_id") == LOCAL_JOB_ID:
        os.environ["SLURM_JOB_ID"] = LOCAL_JOB_ID
    trainer = load_trainer()
    sys.argv = [trainer.__file__, *argv]
    return trainer.main()


def run_worker_server(argv):
    from native_runner.training.v4 import remote_ppo_worker_server

    sys.argv = [remote_ppo_worker_server.__file__, *argv]
    return remote_ppo_worker_server.main()


def probe_engines(count, base_port):
    """Ask each running engine (e.g. from upstream start_engine_cluster.ps1) for its status and attestation."""
    from native_runner.training.v4.configure_emulator_engine_guests import control

    return [{"port": base_port + index, "status": control(base_port + index, "engine-status", timeout=3.0),
             "attestation": control(base_port + index, "attest", timeout=8.0).get("attestation", {})} for index in range(count)]


def local_topology(endpoints, *, worker_port, worker_cpus, control_cpus):
    """Upstream's topology for one PC, after the acceptance checks of launch_ppo_engine_node.sh minus its Slurm check."""
    from native_runner.training.v4.build_ppo_multinode_topology import build_topology

    checks = {
        "engines_answer": bool(endpoints) and all(row["status"].get("ok") is True for row in endpoints),
        "engine_ids": [row["status"].get("engineId") for row in endpoints] == list(range(len(endpoints))),
        "unique_processes": len({row["status"].get("pid") for row in endpoints}) == len(endpoints),
        "resident_slots": all(int(row["status"].get("slotsPerEngine", -1)) == 8 for row in endpoints),
        "production_ready": all(row["attestation"].get("production_ready") is True for row in endpoints),
        "one_attestation": len({row["attestation"].get("attestation_digest") for row in endpoints}) == 1,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"engine acceptance failed: {failed}")
    manifest = {"ok": True, "remote_worker_host": "127.0.0.1", "remote_worker_port": worker_port,
                "engines": [{"engine_index": index, "host": "127.0.0.1", "port": row["port"]} for index, row in enumerate(endpoints)],
                "attestation_digest": endpoints[0]["attestation"]["attestation_digest"], "decision_ticks": 5, "resident_slots_per_engine": 8}
    return build_topology([manifest], trainer_job_id=LOCAL_JOB_ID, trainer_host=socket.gethostname().split(".")[0],
                          gpu_count=1, worker_cpu_pool=worker_cpus, control_cpus=control_cpus)


def topology_command(args):
    from native_runner.training.v4.build_ppo_multinode_topology import parse_cpu_set

    try:
        endpoints = probe_engines(args.engines, args.engine_base_port)
    except OSError as error:
        print(f"No engine answers on 127.0.0.1:{args.engine_base_port}..{args.engine_base_port + args.engines - 1} ({error}); "
              "upstream native_runner/start_engine_cluster.ps1 starts them.", file=sys.stderr)
        return 2
    topology = local_topology(endpoints, worker_port=args.worker_port, control_cpus=parse_cpu_set(args.control_cpus),
                              worker_cpus=parse_cpu_set(args.worker_cpus or f"2-{os.cpu_count() - 1}"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(topology, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: topology[key] for key in ("engine_count", "engine_ports", "engine_worker_ports")}))
    return 0


if WINDOWS:
    _patch_windows()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["trainer"]:
        return run_trainer(argv[1:])
    if argv[:1] == ["worker-server"]:
        return run_worker_server(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("trainer", help="upstream train_ppo_self_play_cluster arguments follow")
    sub.add_parser("worker-server", help="upstream remote_ppo_worker_server arguments follow")
    top = sub.add_parser("topology", help="single-PC topology from engines that are already running")
    top.add_argument("--engines", type=int, required=True, help="engine processes, each with 8 resident slots")
    top.add_argument("--engine-base-port", type=int, default=26789, help="start_engine_cluster.ps1's CR_CONTROL_PORT")
    top.add_argument("--output", type=Path, default=ROOT / "local/topology.json")
    top.add_argument("--worker-port", type=int, default=29900)
    top.add_argument("--worker-cpus", help="default: CPUs 2 and up")
    top.add_argument("--control-cpus", default="0,1")
    return topology_command(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
