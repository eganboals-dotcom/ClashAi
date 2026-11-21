import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from lab import read_metrics, summarize_metrics, config
from train import command_for


class MetricsTests(unittest.TestCase):
    def test_resume_deduplicates_updates_and_ignores_partial_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            rows = [{"stage": "ppo", "update": 1, "completed_games_this_segment": 8}, {"stage": "ppo", "update": 1, "completed_games_this_segment": 6}, {"stage": "ppo", "update": 2, "completed_games_this_segment": 4}]
            path.write_text("\n".join(json.dumps(r) for r in rows) + '\n{"stage":', encoding="utf-8")
            result = summarize_metrics(read_metrics(path), 3600)
            self.assertEqual(result["completed_games"], 10)
            self.assertEqual(result["projected_games_seven_days"], 1680)

    def test_missing_metrics_does_not_invent_games(self):
        self.assertEqual(summarize_metrics([])["completed_games"], 0)

    def test_invalid_elapsed_rejected(self):
        for value in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                summarize_metrics([], value)

    def test_generalist_has_no_fixed_learner_deck(self):
        cmd = command_for(config(), Path("output"), 100)
        self.assertNotIn("--learner-deck-id", cmd)
        self.assertEqual(cmd[cmd.index("--self-play-percent") + 1], "50")
        self.assertEqual(cmd.count("--fixed-opponent-checkpoint"), 3)

    def test_resume_is_explicit(self):
        cmd = command_for(config(), Path("new-attempt"), 100, Path("strict.pt"))
        self.assertEqual(cmd[-2:], ["--resume-checkpoint", "strict.pt"])


if __name__ == "__main__":
    unittest.main()
