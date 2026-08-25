"""DB-free tests for the ingestion content hash and the skip decision."""

import hashlib

from app.ingest import build_card_content, compute_content_hash, should_skip_card


def _card(**overrides):
    card = {
        "id": 4007,
        "name": "Magician of Black Chaos",
        "englishAttribute": "dark",
        "level": 8,
        "atk": 2800,
        "def": 2600,
        "properties": ["Spellcaster", "Ritual"],
        "effectText": "This monster can only be Ritual Summoned.",
    }
    card.update(overrides)
    return card


# --- compute_content_hash -------------------------------------------------


def test_same_content_same_hash():
    assert compute_content_hash("Name: Kuriboh") == compute_content_hash("Name: Kuriboh")


def test_changed_content_different_hash():
    assert compute_content_hash("Name: Kuriboh") != compute_content_hash("Name: Kuribah")


def test_hash_is_sha256_hexdigest_of_the_content():
    content = "Name: Dark Magician | ATK: 2500"
    assert compute_content_hash(content) == hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()


def test_hash_is_64_lowercase_hex_chars():
    digest = compute_content_hash("Name: Dark Magician")
    assert len(digest) == 64
    assert digest == digest.lower()
    assert all(c in "0123456789abcdef" for c in digest)


def test_empty_content_still_hashes():
    assert len(compute_content_hash("")) == 64


def test_hash_tracks_build_card_content():
    # Two identical cards hash the same through the real content builder.
    a = compute_content_hash(build_card_content(_card()))
    b = compute_content_hash(build_card_content(_card()))
    assert a == b


def test_edited_effect_text_changes_the_hash():
    # The whole point of hashing content rather than keying on card id: an
    # edited JSON must hash differently so it gets re-ingested.
    before = compute_content_hash(build_card_content(_card()))
    after = compute_content_hash(
        build_card_content(_card(effectText="Errata: cannot be Special Summoned."))
    )
    assert before != after


def test_edited_atk_changes_the_hash():
    before = compute_content_hash(build_card_content(_card()))
    after = compute_content_hash(build_card_content(_card(atk=3000)))
    assert before != after


# --- should_skip_card -----------------------------------------------------


def test_skip_when_stored_hash_matches():
    digest = compute_content_hash(build_card_content(_card()))
    assert should_skip_card(4007, digest, {4007: digest}) is True


def test_no_skip_when_stored_hash_differs():
    # Same card id, edited content: must NOT be skipped.
    stored = compute_content_hash(build_card_content(_card()))
    fresh = compute_content_hash(build_card_content(_card(atk=3000)))
    assert should_skip_card(4007, fresh, {4007: stored}) is False


def test_no_skip_when_card_absent_from_db():
    digest = compute_content_hash(build_card_content(_card()))
    assert should_skip_card(4007, digest, {}) is False


def test_no_skip_when_stored_hash_is_null():
    # Rows ingested before the column existed carry NULL and are stale.
    digest = compute_content_hash(build_card_content(_card()))
    assert should_skip_card(4007, digest, {4007: None}) is False


def test_no_skip_when_card_json_id_missing():
    digest = compute_content_hash(build_card_content(_card()))
    assert should_skip_card(None, digest, {4007: digest}) is False


def test_skip_decision_is_per_card_id():
    # A matching hash under a different id does not license a skip.
    digest = compute_content_hash(build_card_content(_card()))
    assert should_skip_card(4007, digest, {1234: digest}) is False


def test_unchanged_then_edited_round_trip():
    # Simulates the resume path: first run stores the hash, second run skips,
    # an edit then breaks the match and the card is re-ingested.
    card = _card()
    stored = {card["id"]: compute_content_hash(build_card_content(card))}

    unchanged = compute_content_hash(build_card_content(_card()))
    assert should_skip_card(card["id"], unchanged, stored) is True

    edited = compute_content_hash(build_card_content(_card(name="Dark Magician")))
    assert should_skip_card(card["id"], edited, stored) is False
