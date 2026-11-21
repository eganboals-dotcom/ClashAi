"""Build an explicitly limited 20-deck smoke-test pool from upstream examples."""
import json
from lab import ROOT, UPSTREAM, save_json
from native_runner.training.v4.deck_pool import DeckPoolManifestV4, deck_identity_from_keys


def main():
    source = UPSTREAM / "native_runner/training/v4/focus_decks_20.json"
    cards = json.loads((UPSTREAM / "native_runner/data/competitive/card_support.json").read_text())["supported_cards"]
    allowed = {int(c["card_id"]): int(c["form_availability"]) for c in cards}
    rows = json.loads(source.read_text())["decks"]
    decks = []
    for row in rows:
        deck = deck_identity_from_keys(row["card_keys"], row["tower_key"], uses=1, focus_rank=int(row["rank"]))
        for card, form in zip(deck.card_ids, deck.form_availability, strict=True):
            if card not in allowed or form & ~allowed[card]:
                raise ValueError(f"Unsupported card/form in upstream starter deck {row['rank']}: {card}/{form}")
        decks.append(deck)
    manifest = DeckPoolManifestV4(tuple(decks), minimum_uses=1, metadata={"purpose": "smoke-test-only", "source": "upstream focus_decks_20.json", "usage_counts_are_measured": False})
    target = ROOT / "local/starter-deck-pool.json"
    manifest.save(target)
    first, second = decks[:2]
    save_json(ROOT / "local/example-match.json", {
        "deck0": list(first.card_ids), "deck1": list(second.card_ids),
        "deck0_form_availability": list(first.form_availability),
        "deck1_form_availability": list(second.form_availability),
        "tower_troop0_id": first.tower_troop_id, "tower_troop1_id": second.tower_troop_id,
    })
    print(json.dumps({"decks": len(decks), "output": str(target), "scope": "20 upstream examples only; replace with a broad replay-derived pool before extended training"}, indent=2))


if __name__ == "__main__":
    main()
