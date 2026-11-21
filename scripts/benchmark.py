"""Per-matchup offline benchmark of a candidate checkpoint against frozen references.

Each matchup is played in both deck orientations. Upstream's evaluator also swaps
engine owners on alternate games and gives both sides the same decision timing.
Results are reported per matchup so one favorable deck cannot hide a regression.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

from lab import ROOT, UPSTREAM, config, save_json

SUITE_SCHEMA = "clashai-benchmark-suite.v1"
COUNTED_EVENTS = ("evolution_played", "ability_activation", "runtime_ability_activation")
# Upstream history: most wins over a passive IL came from games where it made <= 3 plays.
PASSIVE_MAX_ACTIONS = 3
# A promoted checkpoint may not lose more than this on its worst matchup.
WORST_MATCHUP_TOLERANCE = 0.05
GIANT, GRAVEYARD = 26000003, 28000010


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_suite(path, pool=None):
    """Resolve every deck through upstream and the pinned catalog; label pool membership."""
    from native_runner.training.v4.deck_pool import DeckPoolManifestV4, deck_identity_from_keys

    suite = json.loads(Path(path).read_text(encoding="utf-8"))
    if suite.get("schema") != SUITE_SCHEMA:
        raise ValueError(f"{path}: expected schema {SUITE_SCHEMA}")
    catalog = json.loads((UPSTREAM / "native_runner/data/competitive/card_support.json").read_text(encoding="utf-8"))
    allowed = {int(c["card_id"]): int(c["form_availability"]) for c in catalog["supported_cards"]}
    pool_ids = None if pool is None else {d.deck_id for d in DeckPoolManifestV4.load(pool).decks}
    decks = {}
    for name, row in suite["decks"].items():
        deck = deck_identity_from_keys(row["cards"], row["tower"], uses=1)
        for card, form in zip(deck.card_ids, deck.form_availability, strict=True):
            if card not in allowed or form & ~allowed[card]:
                raise ValueError(f"deck {name}: card/form {card}/{form} is outside the pinned catalog")
        decks[name] = {
            "identity": deck,
            "source": row.get("source", "unspecified"),
            "in_pool": None if pool_ids is None else deck.deck_id in pool_ids,
            "evolution_slots": sum(form == 1 for form in deck.form_availability),
            "hero_slots": sum(form == 2 for form in deck.form_availability),
        }
    matchups = [tuple(pair) for pair in suite["matchups"]]
    for a, b in matchups:
        if a not in decks or b not in decks or a == b:
            raise ValueError(f"invalid matchup {a} vs {b}")
    if len(set(matchups)) != len(matchups):
        raise ValueError("suite repeats a matchup")
    return decks, matchups, canonical_hash(suite)


def match_config(deck0, deck1, seed):
    return {
        "deck0": list(deck0.card_ids), "deck1": list(deck1.card_ids),
        "deck0_form_availability": list(deck0.form_availability),
        "deck1_form_availability": list(deck1.form_availability),
        "tower_troop0_id": deck0.tower_troop_id, "tower_troop1_id": deck1.tower_troop_id,
        "seed": seed,
    }


def wilson(score, games, z=1.96):
    if games == 0:
        return [0.0, 1.0]
    denominator = 1 + z * z / games
    center = (score + z * z / (2 * games)) / denominator
    half = z * math.sqrt(score * (1 - score) / games + z * z / (4 * games * games)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def enrich(row, tally):
    """Re-key per-owner counts to the candidate ("a") and reference ("b") policies."""
    owners = {"candidate": row["a_owner"], "reference": 1 - row["a_owner"]}
    for role, owner in owners.items():
        row[role] = {
            "actions": row["actions_by_owner"][owner],
            "evolutions": tally["events"][(owner, "evolution_played")],
            "abilities": tally["events"][(owner, "ability_activation")] + tally["events"][(owner, "runtime_ability_activation")],
        }
    row["elapsed_seconds"] = tally["ended"] - tally["started"]
    return row


def summarize(rows):
    games = len(rows)
    wins = sum(r["winner"] == "a" for r in rows)
    losses = sum(r["winner"] == "b" for r in rows)
    draws = games - wins - losses
    score = (wins + 0.5 * draws) / games if games else 0.0
    elapsed = sum(r["elapsed_seconds"] for r in rows)
    total = lambda role, key: sum(r[role][key] for r in rows)
    return {
        "games": games, "wins": wins, "losses": losses, "draws": draws,
        "truncated": sum(bool(r["truncated"]) for r in rows),
        "rejected_actions": sum(r["rejected_actions"] for r in rows),
        "score": score, "ci95": wilson(score, games),
        "wins_vs_passive_reference": sum(r["winner"] == "a" and r["reference"]["actions"] <= PASSIVE_MAX_ACTIONS for r in rows),
        "passive_candidate_games": sum(r["candidate"]["actions"] <= PASSIVE_MAX_ACTIONS for r in rows),
        "candidate_evolutions": total("candidate", "evolutions"), "candidate_abilities": total("candidate", "abilities"),
        "reference_evolutions": total("reference", "evolutions"), "reference_abilities": total("reference", "abilities"),
        "elapsed_seconds": elapsed, "games_per_hour": games * 3600 / elapsed if elapsed > 0 else 0.0,
    }


def build_report(results, decks, identity):
    """results: {(reference, matchup_name): [game rows from both orientations]}."""
    references = {}
    for (reference, name), rows in sorted(results.items()):
        summary = summarize(rows)
        a, b = name.split(" vs ")
        summary["held_out"] = None if decks[a]["in_pool"] is None else not (decks[a]["in_pool"] and decks[b]["in_pool"])
        references.setdefault(reference, {})[name] = summary
    per_reference = {}
    for reference, matchups in references.items():
        scores = {name: s["score"] for name, s in matchups.items()}
        worst = min(scores, key=scores.get)
        held = [s["score"] for s in matchups.values() if s["held_out"]]
        per_reference[reference] = {
            "macro_score": sum(scores.values()) / len(scores),
            "worst_matchup": worst, "worst_matchup_score": scores[worst],
            "matchups_significantly_below_half": sorted(n for n, s in matchups.items() if s["ci95"][1] < 0.5),
            "held_out_macro_score": sum(held) / len(held) if held else None,
            "matchups": matchups,
        }
    every = [s for matchups in references.values() for s in matchups.values()]
    elapsed = sum(s["elapsed_seconds"] for s in every)
    return {
        **identity,
        "overall_score": sum(r["macro_score"] for r in per_reference.values()) / len(per_reference),
        "worst_matchup_score": min(r["worst_matchup_score"] for r in per_reference.values()),
        "games": sum(s["games"] for s in every),
        "rejected_actions": sum(s["rejected_actions"] for s in every),
        "matchups_with_evolution_and_ability": sorted({
            f"{ref}: {name}" for ref, matchups in references.items() for name, s in matchups.items()
            if s["candidate_evolutions"] + s["reference_evolutions"] and s["candidate_abilities"] + s["reference_abilities"]
        }),
        "elapsed_seconds": elapsed, "games_per_hour": sum(s["games"] for s in every) * 3600 / elapsed if elapsed > 0 else 0.0,
        "references": per_reference,
        "note": "Model-versus-model offline results on this suite only; not a ladder rating.",
    }


def promotion_decision(report, incumbent):
    if report["rejected_actions"]:
        return False, "native actions were rejected"
    if incumbent is None:
        return True, "no incumbent"
    if incumbent["comparable_key"] != report["comparable_key"]:
        return False, "incumbent was measured with a different suite, references, games or seed and needs a fresh benchmark"
    if report["overall_score"] <= incumbent["overall_score"]:
        return False, "overall score is not higher than the incumbent's"
    if report["worst_matchup_score"] < incumbent["worst_matchup_score"] - WORST_MATCHUP_TOLERANCE:
        return False, "worst matchup regressed beyond tolerance"
    return True, "higher overall score without a worst-matchup regression"


def promote(report, candidate, report_path, best):
    best.mkdir(parents=True, exist_ok=True)
    incumbent = json.loads((best / "best.json").read_text()) if (best / "best.json").is_file() else None
    accepted, reason = promotion_decision(report, incumbent)
    if accepted:
        digest = report["candidate"]["sha256"]
        target = best / f"checkpoint-{digest[:12]}.pt"
        if not target.exists():
            shutil.copy2(candidate, target)
        save_json(best / "best.json", {
            "checkpoint": str(target), "sha256": digest, "source": str(candidate), "report": str(report_path),
            "overall_score": report["overall_score"], "worst_matchup_score": report["worst_matchup_score"],
            "comparable_key": report["comparable_key"], "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
    return {"promoted": accepted, "reason": reason}


def play(candidate, reference, config_row, games, output, host, port, device):
    """Run upstream's evaluate() unchanged while tallying per-owner form/ability events."""
    import native_runner.training.v4.evaluate as upstream
    from native_runner.match_factory import MatchConfig

    tallies = []

    class TallyEnv(upstream.BattleEnvV1):
        def reset(self, *args, **kwargs):
            tallies.append({"started": time.monotonic(), "ended": time.monotonic(), "seen": set(), "events": Counter()})
            return super().reset(*args, **kwargs)

        def step(self, *args, **kwargs):
            result = super().step(*args, **kwargs)
            tally = tallies[-1]
            for observation in result[0].values():
                for event in observation.events:
                    key = (event.tick, event.event_type, event.owner, event.entity_id, event.card_id)
                    if event.event_type in COUNTED_EVENTS and event.owner is not None and key not in tally["seen"]:
                        tally["seen"].add(key)
                        tally["events"][(event.owner, event.event_type)] += 1
            tally["ended"] = time.monotonic()
            return result

    original, upstream.BattleEnvV1 = upstream.BattleEnvV1, TallyEnv
    try:
        upstream.evaluate(candidate, reference, games=games, match=MatchConfig(**config_row), output=output, host=host, port=port, device=device)
    finally:
        upstream.BattleEnvV1 = original
    rows = [json.loads(line) for line in (Path(output) / "matches.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != len(tallies):
        raise RuntimeError("game log and event tallies disagree")
    rows = [enrich(row, tally) for row, tally in zip(rows, tallies)]
    (Path(output) / "games.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return rows


def engine_unreachable(error):
    """Upstream wraps socket failures in RunnerError/BattleEnvError; look for the OSError cause."""
    while error is not None:
        if isinstance(error, OSError):
            return True
        error = error.__cause__ or error.__context__
    return False


def reference_labels(paths):
    labels = {}
    for path in paths:
        label = path.parent.name if path.stem.startswith("checkpoint") else path.stem
        label = label.replace(" ", "-")
        while label in labels.values():
            label += "+"
        labels[path] = label
    return labels


def write_markdown(report, path):
    lines = [f"# Benchmark: {Path(report['candidate']['path']).name}", "",
             f"Overall {report['overall_score']:.3f}; worst matchup {report['worst_matchup_score']:.3f}; "
             f"{report['games']} games; {report['rejected_actions']} rejected actions; {report['games_per_hour']:.1f} games/hour.", ""]
    for reference, data in report["references"].items():
        lines += [f"## vs {reference}: macro {data['macro_score']:.3f}", "",
                  "| Matchup | Held out | Score | 95% CI | W-L-D | Evo/abil (cand) | Passive-ref wins | Rejected |",
                  "| --- | --- | ---: | --- | --- | --- | ---: | ---: |"]
        for name, s in data["matchups"].items():
            held = {None: "unknown", True: "yes", False: "no"}[s["held_out"]]
            lines.append(f"| {name} | {held} | {s['score']:.3f} | {s['ci95'][0]:.2f}–{s['ci95'][1]:.2f} | "
                         f"{s['wins']}-{s['losses']}-{s['draws']} | {s['candidate_evolutions']}/{s['candidate_abilities']} | "
                         f"{s['wins_vs_passive_reference']} | {s['rejected_actions']} |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args):
    cfg = config()
    candidate = Path(args.candidate).resolve() if args.candidate else ROOT / cfg["checkpoint"]
    references = [Path(p).resolve() for p in args.reference] if args.reference else [ROOT / p for p in cfg["opponents"]]
    for path in (candidate, *references):
        if not path.is_file():
            raise FileNotFoundError(path)
    decks, matchups, suite_hash = load_suite(args.suite, args.pool)
    for name in filter(None, (args.only, args.candidate_deck)):
        matchups = [m for m in matchups if name in m]
        if not matchups:
            raise ValueError(f"no matchup uses deck {name}")
    labels = reference_labels(references)
    identity = {
        "candidate": {"path": str(candidate), "sha256": sha256(candidate)},
        "references": {labels[p]: sha256(p) for p in references},
        "suite_sha256": suite_hash, "games_per_orientation": args.games, "seed": args.seed,
        "candidate_deck": args.candidate_deck,
        "upstream_commit": cfg["upstream_commit"], "decision": "argmax",
    }
    identity["comparable_key"] = canonical_hash({k: identity[k] for k in ("references", "suite_sha256", "games_per_orientation", "seed", "candidate_deck", "upstream_commit")} | {"matchups": matchups})
    # Each suite/deck combination keeps its own best checkpoint, so a specialist never competes with the generalist.
    track = Path(args.suite).stem + (f"-{args.candidate_deck}" if args.candidate_deck else "")
    identity["reference_paths"] = {labels[p]: str(p) for p in references}
    out = ROOT / "runs/benchmarks" / (args.name or time.strftime("benchmark-%Y%m%d-%H%M%S"))
    out.mkdir(parents=True, exist_ok=False)
    plan = [{"reference": labels[r], "reference_path": str(r), "matchup": f"{a} vs {b}", "orientation": orient, "candidate_deck": x, "reference_deck": y,
             "match_config": match_config(decks[x]["identity"], decks[y]["identity"], args.seed)}
            for r in references for a, b in matchups for orient, (x, y) in (("forward", (a, b)), ("reverse", (b, a)))
            if args.candidate_deck in (None, x)]
    save_json(out / "plan.json", {**identity, "runs": plan})
    if args.dry_run:
        print(json.dumps({"output": str(out), "runs": len(plan), "games": len(plan) * args.games, "dry_run": True}, indent=2))
        return 0
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(1)
    results = {}
    for item in plan:
        leaf = out / item["reference"] / item["matchup"].replace(" ", "_") / item["orientation"]
        try:
            rows = play(candidate, Path(item["reference_path"]), item["match_config"], args.games, leaf, args.host, args.port, device)
        except Exception as error:
            if not engine_unreachable(error):
                raise
            print(f"No offline engine is reachable at {args.host}:{args.port} ({error}); docs/SETUP.md describes starting one.", file=sys.stderr)
            return 2
        results.setdefault((item["reference"], item["matchup"]), []).extend(rows)
        if any(r["rejected_actions"] for r in rows):
            print(f"Stopping: native actions were rejected in {leaf}", file=sys.stderr)
            break
    report = build_report(results, decks, identity)
    save_json(out / "report.json", report)
    write_markdown(report, out / "report.md")
    if args.promote:
        report["promotion"] = promote(report, candidate, out / "report.json", ROOT / "runs/best" / track)
        save_json(out / "report.json", report)
    print(json.dumps({k: report[k] for k in ("overall_score", "worst_matchup_score", "games", "rejected_actions", "games_per_hour")} | {"report": str(out / "report.md"), "promotion": report.get("promotion")}, indent=2))
    return 0 if not report["rejected_actions"] else 1


def is_giant_graveyard(deck):
    return GIANT in deck.card_ids and GRAVEYARD in deck.card_ids


def entry(deck, source):
    return {"tower": deck.tower_key, "source": source, "cards": list(deck.card_keys)}


def distinct(decks, count):
    """Most-used first, skipping Giant Graveyard decks and near-duplicates (six or more shared cards)."""
    chosen = []
    for deck in sorted(decks, key=lambda d: (-d.uses, d.deck_id)):
        if len(chosen) == count:
            break
        if not is_giant_graveyard(deck) and all(len(set(deck.card_ids) & set(c.card_ids)) < 6 for c in chosen):
            chosen.append(deck)
    return chosen


def load_pools(pool_path, all_path):
    from native_runner.training.v4.deck_pool import DeckPoolManifestV4

    pool = DeckPoolManifestV4.load(pool_path).decks
    pool_ids = {d.deck_id for d in pool}
    unseen = sorted((d for d in DeckPoolManifestV4.load(all_path).decks if d.deck_id not in pool_ids), key=lambda d: (-d.uses, d.deck_id))
    return pool, unseen


def build_suite(args):
    """Add replay-derived Giant Graveyard and held-out decks to the starter suite."""
    base = json.loads(Path(args.suite).read_text(encoding="utf-8"))
    pool, unseen = load_pools(args.pool, args.all_decks)
    added = {}
    trained_gy = sorted((d for d in pool if is_giant_graveyard(d)), key=lambda d: (-d.uses, d.deck_id))
    if trained_gy:
        added["giant-graveyard-replay"] = entry(trained_gy[0], f"replay pool, {trained_gy[0].uses} uses")
    held_gy = [d for d in unseen if is_giant_graveyard(d)]
    if held_gy:
        added["giant-graveyard-held-out"] = entry(held_gy[0], f"replay, {held_gy[0].uses} uses, not in training pool")
    chosen = distinct(unseen, args.held_out)
    for index, deck in enumerate(chosen, 1):
        added[f"held-out-{index}"] = entry(deck, f"replay, {deck.uses} uses, not in training pool")
    gy_names = [n for n in ("giant-graveyard-replay", "giant-graveyard-held-out") if n in added]
    held_names = [f"held-out-{i}" for i in range(1, len(chosen) + 1)]
    matchups = [list(m) for m in base["matchups"]]
    matchups += [[gy, h] for gy in gy_names for h in held_names]
    matchups += [[held_names[i], held_names[i + 1]] for i in range(0, len(held_names) - 1, 2)]
    suite = {**base, "decks": {**base["decks"], **added}, "matchups": matchups,
             "note": base["note"] + f" Generated from {Path(args.pool).name} and {Path(args.all_decks).name}."}
    save_json(args.output, suite)
    load_suite(args.output, args.pool)
    print(json.dumps({"output": str(args.output), "added_decks": sorted(added), "matchups": len(matchups),
                      "giant_graveyard_in_pool": len(trained_gy), "giant_graveyard_held_out": len(held_gy)}, indent=2))
    return 0


SPECIALIST_DECK = "giant-graveyard"


def specialist_suite(args):
    """The training config's fixed learner deck against focus, common, held-out and other Giant Graveyard decks."""
    from native_runner.training.v4.deck_pool import deck_identity_from_keys

    learner_row = config(args.config)["learner_deck"]
    learner = deck_identity_from_keys(learner_row["cards"], learner_row["tower"], uses=1)
    pool, unseen = load_pools(args.pool, args.all_decks)
    trained = [d for d in pool if d.deck_id != learner.deck_id]
    opponents = {f"focus-{d.focus_rank}": entry(d, f"upstream focus rank {d.focus_rank}, {d.uses} uses")
                 for d in sorted((d for d in trained if d.focus_rank), key=lambda d: d.focus_rank)}
    opponents |= {f"common-{i}": entry(d, f"replay pool, {d.uses} uses")
                  for i, d in enumerate(distinct((d for d in trained if not d.focus_rank), args.common), 1)}
    opponents |= {f"held-out-{i}": entry(d, f"replay, {d.uses} uses, not in training pool")
                  for i, d in enumerate(distinct(unseen, args.held_out), 1)}
    other_gy = sorted((d for d in trained if is_giant_graveyard(d)), key=lambda d: (-d.uses, d.deck_id))
    if other_gy:
        opponents["giant-graveyard-other"] = entry(other_gy[0], f"replay pool, {other_gy[0].uses} uses")
    held_gy = [d for d in unseen if is_giant_graveyard(d)]
    if held_gy:
        opponents["giant-graveyard-held-out"] = entry(held_gy[0], f"replay, {held_gy[0].uses} uses, not in training pool")
    suite = {
        "schema": SUITE_SCHEMA,
        "note": f"Specialist suite: the learner deck from {Path(args.config).name} plays every opponent deck. Run with --candidate-deck {SPECIALIST_DECK}.",
        "decks": {SPECIALIST_DECK: {"tower": learner_row["tower"], "source": "training config learner deck", "cards": list(learner.card_keys)}, **opponents},
        "matchups": [[SPECIALIST_DECK, name] for name in opponents],
    }
    save_json(args.output, suite)
    decks, matchups, _ = load_suite(args.output, args.pool)
    print(json.dumps({"output": str(args.output), "learner_in_pool": decks[SPECIALIST_DECK]["in_pool"], "opponent_decks": len(opponents),
                      "held_out_opponents": sum(not d["in_pool"] for d in decks.values())}, indent=2))
    return 0


def validate(args):
    decks, matchups, digest = load_suite(args.suite, args.pool)
    rows = {name: {k: v for k, v in d.items() if k != "identity"} | {"tower": d["identity"].tower_key, "deck_id": d["identity"].deck_id[:16]} for name, d in decks.items()}
    print(json.dumps({"suite_sha256": digest, "decks": rows, "matchups": len(matchups)}, indent=2))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "run", "build-suite", "specialist-suite"):
        p = sub.add_parser(name)
        p.add_argument("--suite", default=str(ROOT / "configs/benchmark-suite.json"))
        p.add_argument("--pool", help="training deck pool; labels decks in/out of training")
    s = sub.choices["specialist-suite"]
    s.add_argument("--config", default=str(ROOT / "configs/giant-graveyard.json"), help="training config holding learner_deck")
    s.add_argument("--all-decks", required=True, help="pool built with --minimum-uses 1")
    s.add_argument("--common", type=int, default=10, help="distinct most-used non-focus pool decks")
    s.add_argument("--held-out", type=int, default=10)
    s.add_argument("--output", default=str(ROOT / "local/giant-graveyard-suite.json"))
    r = sub.choices["run"]
    r.add_argument("--candidate", help="checkpoint to test; defaults to General")
    r.add_argument("--reference", action="append", help="repeatable frozen reference; defaults to the configured opponents")
    r.add_argument("--games", type=int, default=10, help="games per deck orientation; even so owners balance")
    r.add_argument("--only", help="limit to matchups using this deck name")
    r.add_argument("--candidate-deck", help="the candidate plays only this deck (specialist mode); one orientation per matchup")
    r.add_argument("--seed", type=int, default=20261120)
    r.add_argument("--name")
    r.add_argument("--host", default="127.0.0.1")
    r.add_argument("--port", type=int, default=26789)
    r.add_argument("--promote", action="store_true", help="keep the checkpoint in runs/best if it beats the incumbent")
    r.add_argument("--dry-run", action="store_true", help="write the plan without contacting the engine")
    b = sub.choices["build-suite"]
    b.add_argument("--all-decks", required=True, help="pool built with --minimum-uses 1")
    b.add_argument("--held-out", type=int, default=6)
    b.add_argument("--output", default=str(ROOT / "local/benchmark-suite.json"))
    args = parser.parse_args(argv)
    if args.command == "run" and (args.games < 2 or args.games % 2):
        parser.error("games must be a positive even number")
    if args.command in ("build-suite", "specialist-suite") and not args.pool:
        parser.error(f"{args.command} needs --pool")
    return {"validate": validate, "run": run, "build-suite": build_suite, "specialist-suite": specialist_suite}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
