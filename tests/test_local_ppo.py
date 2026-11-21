import multiprocessing
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import local_ppo
from local_ppo import load_trainer, local_topology
from train import command_for, worker_server_arguments
from lab import config


def engine(index, digest="d", ready=True, slots=8):
    return {"port": 26789 + index, "status": {"ok": True, "engineId": index, "pid": 100 + index, "slotsPerEngine": slots},
            "attestation": {"production_ready": ready, "attestation_digest": digest}}


class TopologyTests(unittest.TestCase):
    def test_local_topology_passes_upstream_trainer_checks(self):
        trainer = load_trainer()
        topology = local_topology([engine(0), engine(1)], worker_port=29900, worker_cpus=(2, 3), control_cpus=(0, 1))
        self.assertEqual(topology["trainer_slurm_job_id"], local_ppo.LOCAL_JOB_ID)
        self.assertEqual(topology["rank_engine_indices"], [[0, 1]])
        # The trainer's own readers accept it.
        self.assertEqual(trainer._ready_engine_ports(topology, engine_count=2), (26789, 26790))
        self.assertEqual(trainer._ready_rank_engine_indices(topology, engine_count=2, world_size=1), ((0, 1),))

    def test_engines_that_are_not_ready_or_disagree_are_refused(self):
        for bad in (engine(1, ready=False), engine(1, digest="other"), engine(1, slots=4)):
            with self.assertRaises(RuntimeError):
                local_topology([engine(0), bad], worker_port=29900, worker_cpus=(2,), control_cpus=(0,))

    def test_worker_service_owns_exactly_the_topology_engines(self):
        topology = local_topology([engine(0), engine(1)], worker_port=29900, worker_cpus=(2,), control_cpus=(0,))
        args = worker_server_arguments(topology)
        base, count = int(args[args.index("--engine-base-port") + 1]), int(args[args.index("--engine-count") + 1])
        # remote_ppo_worker_server computes local_index = endpoint_port - (engine_base_port + global_start).
        self.assertEqual([port - base for port in topology["engine_ports"]], list(range(count)))


class TrainerTests(unittest.TestCase):
    def test_trainer_accepts_the_public_pool_size_only_through_the_focus_check(self):
        source = Path(load_trainer().__file__).read_text(encoding="utf-8")
        self.assertEqual(source.count(local_ppo.POOL_CHECK), 1)  # pinned upstream unchanged on disk
        self.assertIn("main", vars(load_trainer()))

    def test_supervisor_runs_the_local_launcher_without_torchrun(self):
        cmd = command_for(config(), Path("out"), 1)
        self.assertEqual(cmd[1:3], [str(local_ppo.ROOT / "scripts/local_ppo.py"), "trainer"])
        self.assertNotIn("torch.distributed.run", cmd)


@unittest.skipUnless(sys.platform == "win32", "Windows shims only")
class WindowsShimTests(unittest.TestCase):
    def test_linux_calls_have_windows_equivalents(self):
        from native_runner.training.v4 import async_cluster_self_play

        self.assertEqual(multiprocessing.get_context("fork").get_start_method(), "spawn")
        self.assertEqual(os.sched_getaffinity(0), set(range(os.cpu_count())))
        self.assertEqual(async_cluster_self_play.ctypes.CDLL(None).prctl(1, 15), 0)


if __name__ == "__main__":
    unittest.main()
