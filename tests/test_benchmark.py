from argparse import Namespace
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from benchmark import (ROOT, build_report, build_suite, config, engine_unreachable, enrich, load_suite, main,
                       promotion_decision, specialist_suite, summarize, wilson)
from native_runner.training.v4.deck_pool import DeckPoolManifestV4, deck_identity_from_keys

SUITE = ROOT / "configs/benchmark-suite.json"


def game(winner, a_owner=0, cand_actions=10, ref_actions=10, rejected=0):
    row = {"a_owner": a_owner, "winner": winner, "truncated": False, "rejected_actions": rejected,
           "actions_by_owner": [cand_actions, ref_actions] if a_owner == 0 else [ref_actions, cand_actions]}
    return enrich(row, {"started": 0.0, "ended": 180.0, "events": Counter({(a_owner, "evolution_played"): 1, (1 - a_owner, "ability_activation"): 2})})


class SuiteTests(unittest.TestCase):
    def test_default_suite_resolves_and_giant_graveyard_has_evolution_and_hero(self):
        decks, matchups, _ = load_suite(SUITE)
        gy = decks["giant-graveyard"]
        self.assertIn(26000003, gy["identity"].card_ids)
        self.assertIn(28000010, gy["identity"].card_ids)
        self.assertEqual((gy["evolution_slots"], gy["hero_slots"]), (2, 1))
        self.assertTrue(any("giant-graveyard" in pair for pair in matchups))
        self.assertGreaterEqual(len({d["identity"].tower_key for d in decks.values()}), 4)

    def test_build_suite_adds_only_decks_outside_the_training_pool(self):
        focus = json.loads((ROOT / "vendor/FirstLight_CR/native_runner/training/v4/focus_decks_20.json").read_text())["decks"]
        trained = tuple(deck_identity_from_keys(r["card_keys"], r["tower_key"], uses=9) for r in focus)
        gy = deck_identity_from_keys(["giant", "graveyard", "poison", "the-log", "knight", "mega-minion", "ice-wizard", "zap"], "tower-princess", uses=3)
        other = deck_identity_from_keys(["hog-rider", "musketeer", "fireball", "the-log", "ice-spirit", "skeletons", "cannon", "ice-golem"], "tower-princess", uses=2)
        with tempfile.TemporaryDirectory() as directory:
            pool, every, out = (Path(directory) / name for name in ("pool.json", "all.json", "suite.json"))
            DeckPoolManifestV4(trained, minimum_uses=5, metadata={}).save(pool)
            DeckPoolManifestV4(trained + (gy, other), minimum_uses=1, metadata={}).save(every)
            build_suite(Namespace(suite=str(SUITE), pool=str(pool), all_decks=str(every), held_out=6, output=str(out)))
            decks, matchups, _ = load_suite(out, pool)
        self.assertNotIn("giant-graveyard-replay", decks)
        self.assertFalse(decks["giant-graveyard-held-out"]["in_pool"])
        self.assertFalse(decks["held-out-1"]["in_pool"])
        self.assertIn(("giant-graveyard-held-out", "held-out-1"), matchups)


class SpecialistTests(unittest.TestCase):
    def setUp(self):
        focus = json.loads((ROOT / "vendor/FirstLight_CR/native_runner/training/v4/focus_decks_20.json").read_text())["decks"]
        self.focus = tuple(deck_identity_from_keys(r["card_keys"], r["tower_key"], uses=9, focus_rank=r["rank"]) for r in focus)
        self.learner_row = json.loads((ROOT / "configs/giant-graveyard.json").read_text())["learner_deck"]
        self.learner = deck_identity_from_keys(self.learner_row["cards"], self.learner_row["tower"], uses=94)
        self.unseen = deck_identity_from_keys(["hog-rider", "musketeer", "fireball", "the-log", "ice-spirit", "skeletons", "cannon", "ice-golem"], "tower-princess", uses=2)

    def test_specialist_suite_pits_the_config_deck_against_every_opponent(self):
        with tempfile.TemporaryDirectory() as directory:
            pool, every, out = (Path(directory) / name for name in ("pool.json", "all.json", "suite.json"))
            DeckPoolManifestV4(self.focus + (self.learner,), minimum_uses=5, metadata={}).save(pool)
            DeckPoolManifestV4(self.focus + (self.learner, self.unseen), minimum_uses=1, metadata={}).save(every)
            specialist_suite(Namespace(config=str(ROOT / "configs/giant-graveyard.json"), pool=str(pool), all_decks=str(every),
                                       common=3, held_out=3, output=str(out)))
            decks, matchups, _ = load_suite(out, pool)
        self.assertTrue(decks["giant-graveyard"]["in_pool"])
        self.assertEqual(decks["giant-graveyard"]["identity"].deck_id, self.learner.deck_id)
        self.assertTrue(all(a == "giant-graveyard" for a, _ in matchups))
        self.assertEqual(len(matchups), 21)  # 20 focus decks + 1 held-out deck
        self.assertFalse(decks["held-out-1"]["in_pool"])

    def test_candidate_deck_mode_plans_only_that_deck(self):
        name = "test-candidate-deck-plan"
        out = ROOT / "runs/benchmarks" / name
        shutil.rmtree(out, ignore_errors=True)
        try:
            main(["run", "--candidate-deck", "giant-graveyard", "--dry-run", "--name", name, "--reference", str(ROOT / config()["checkpoint"])])
            plan = json.loads((out / "plan.json").read_text())
        finally:
            shutil.rmtree(out, ignore_errors=True)
        self.assertEqual({r["candidate_deck"] for r in plan["runs"]}, {"giant-graveyard"})
        self.assertEqual(len(plan["runs"]), 6)  # six starter matchups use giant-graveyard, one orientation each
        self.assertEqual(plan["candidate_deck"], "giant-graveyard")


class ScoringTests(unittest.TestCase):
    def test_enrich_maps_owner_counts_to_policies(self):
        row = game("a", a_owner=1, cand_actions=7, ref_actions=2)
        self.assertEqual(row["candidate"], {"actions": 7, "evolutions": 1, "abilities": 0})
        self.assertEqual(row["reference"], {"actions": 2, "evolutions": 0, "abilities": 2})
        self.assertEqual(row["elapsed_seconds"], 180.0)

    def test_summary_counts_draws_passivity_and_rate(self):
        rows = [game("a"), game("b", a_owner=1), game(None), game("a", ref_actions=1)]
        s = summarize(rows)
        self.assertEqual((s["wins"], s["losses"], s["draws"]), (2, 1, 1))
        self.assertAlmostEqual(s["score"], 0.625)
        self.assertEqual(s["wins_vs_passive_reference"], 1)
        self.assertAlmostEqual(s["games_per_hour"], 20.0)

    def test_wilson_interval_contains_score_and_is_bounded(self):
        low, high = wilson(0.5, 20)
        self.assertTrue(0 < low < 0.5 < high < 1)
        self.assertEqual(wilson(1.0, 4)[1], 1.0)

    def test_report_is_per_matchup_so_one_deck_cannot_hide_a_loss(self):
        decks, _, _ = load_suite(SUITE)
        results = {("General", "giant-graveyard vs hog-2.6"): [game("a")] * 10, ("General", "hog-2.6 vs xbow"): [game("b")] * 10}
        report = build_report(results, decks, {"comparable_key": "k"})
        self.assertAlmostEqual(report["overall_score"], 0.5)
        self.assertEqual(report["worst_matchup_score"], 0.0)
        self.assertEqual(report["references"]["General"]["matchups_significantly_below_half"], ["hog-2.6 vs xbow"])
        self.assertIn("General: giant-graveyard vs hog-2.6", report["matchups_with_evolution_and_ability"])


class PromotionTests(unittest.TestCase):
    base = {"comparable_key": "k", "overall_score": 0.55, "worst_matchup_score": 0.40, "rejected_actions": 0}

    def test_rules(self):
        incumbent = dict(self.base, overall_score=0.50)
        self.assertTrue(promotion_decision(self.base, None)[0])
        self.assertTrue(promotion_decision(self.base, incumbent)[0])
        self.assertFalse(promotion_decision(dict(self.base, rejected_actions=1), incumbent)[0])
        self.assertFalse(promotion_decision(self.base, dict(incumbent, comparable_key="other"))[0])
        self.assertFalse(promotion_decision(dict(self.base, overall_score=0.50), incumbent)[0])
        self.assertFalse(promotion_decision(dict(self.base, worst_matchup_score=0.30), incumbent)[0])

    def test_engine_unreachable_follows_wrapped_causes(self):
        try:
            try:
                raise ConnectionRefusedError("refused")
            except OSError as error:
                raise RuntimeError("attestation failed") from error
        except RuntimeError as wrapped:
            self.assertTrue(engine_unreachable(wrapped))
        self.assertFalse(engine_unreachable(ValueError("bad deck")))


if __name__ == "__main__":
    unittest.main()
