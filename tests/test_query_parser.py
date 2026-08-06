"""DB-free unit tests for ATK constraint extraction from free-text queries."""

from app.query_parser import AtkConstraint, extract_atk_constraint, prepare_search_query


def _bounds(query):
    c = extract_atk_constraint(query)
    return c.min_atk, c.max_atk


# --- EN threshold min forms ------------------------------------------------

def test_en_threshold_min_forms():
    queries = [
        "more than 2500 ATK",
        "over 2500 ATK",
        "above 2500",
        "greater than 2500 ATK",
        "at least 2500",
        "no less than 2500",
        "minimum of 2500",
        "minimum 2500",
        "exceeding 2500 ATK",
        "exceeds 2500",
    ]
    for q in queries:
        assert _bounds(q) == (2500, None), q


# --- EN less-than forms ----------------------------------------------------

def test_en_less_than_forms():
    queries = [
        "less than 2500 ATK",
        "under 2500",
        "below 2500 ATK",
        "at most 2500",
        "no more than 2500",
        "maximum of 2500",
        "2500 or less",
    ]
    for q in queries:
        assert _bounds(q) == (None, 2500), q


# --- PT-BR threshold min forms ----------------------------------------------

def test_pt_br_threshold_min_forms():
    queries = [
        "mais de 2500 de ataque",
        "acima de 2500",
        "maior que 2500 ATK",
        "pelo menos 2500",
        "no mínimo 2500",
        "mínimo de 2500",
        "2500 ou mais",
    ]
    for q in queries:
        assert _bounds(q) == (2500, None), q


# --- PT-BR less-than forms --------------------------------------------------

def test_pt_br_less_than_forms():
    queries = [
        "menos de 2500",
        "abaixo de 2500 ATK",
        "menor que 2500",
        "no máximo 2500",
        "máximo de 2500",
        "2500 ou menos",
    ]
    for q in queries:
        assert _bounds(q) == (None, 2500), q


# --- Operator forms ---------------------------------------------------------

def test_operator_forms():
    cases = [
        (">= 2500", (2500, None)),
        ("> 2500", (2500, None)),
        ("≥ 2500", (2500, None)),
        ("2500+", (2500, None)),
        ("<= 2500", (None, 2500)),
        ("< 2500", (None, 2500)),
        ("≤ 2500", (None, 2500)),
        (">=2500 ATK", (2500, None)),
    ]
    for q, expected in cases:
        assert _bounds(q) == expected, q


# --- Equality forms ---------------------------------------------------------

def test_equality_forms():
    queries = [
        "exactly 2500 ATK",
        "ATK 2500",
        "ATK equal to 2500",
        "ataque igual a 2500",
        "2500 ATK",
        "with 2500 ATK",
        "com 2500 de ataque",
        "ATK de 2500",
        "2500 attack power",
    ]
    for q in queries:
        assert _bounds(q) == (2500, 2500), q


# --- Range forms ------------------------------------------------------------

def test_range_forms():
    queries = [
        "between 1000 and 2000 ATK",
        "entre 1000 e 2000",
        "1000-2000 ATK",
        "1000 to 2000 ATK",
        "1000 a 2000 de ataque",
        "1000 até 2000 ATK",
        "1000 and 2000 ATK",
    ]
    for q in queries:
        assert _bounds(q) == (1000, 2000), q


def test_reversed_range_normalization():
    assert _bounds("between 3000 and 2000") == (2000, 3000)


# --- Precedence -------------------------------------------------------------

def test_precedence_comparative_beats_equality():
    # "more than 2500 ATK" is a threshold, not an equality.
    assert _bounds("more than 2500 ATK") == (2500, None)


def test_precedence_range_beats_single():
    assert _bounds("between 1000 and 2000 ATK") == (1000, 2000)


# --- No-constraint passthrough ----------------------------------------------

def test_no_constraint_passthrough():
    queries = [
        "blue eyes white dragon",
        "3 dragons",
        "dragon level 8",
        "dragon with the highest attack power",
    ]
    for q in queries:
        c = extract_atk_constraint(q)
        assert (c.min_atk, c.max_atk) == (None, None), q
        assert c.remainder_query == q


def test_bare_number_without_keyword_ignored():
    for q in ("2500", "dragons 2500"):
        c = extract_atk_constraint(q)
        assert (c.min_atk, c.max_atk) == (None, None), q
        assert c.remainder_query == q


def test_malformed_not_extracted():
    queries = ["atk abc", "more than abc ATK", "2500.5 ATK", "ATK -2500", "-1 ATK"]
    for q in queries:
        c = extract_atk_constraint(q)
        assert (c.min_atk, c.max_atk) == (None, None), q
        assert c.remainder_query == q


# --- Case / accent insensitivity --------------------------------------------

def test_case_insensitive():
    c = extract_atk_constraint("MORE THAN 2500 ATK")
    assert (c.min_atk, c.max_atk) == (2500, None)

    c = extract_atk_constraint("Dragões Com Mais De 4000 De Ataque")
    assert (c.min_atk, c.max_atk) == (4000, None)


def test_accent_insensitive():
    assert _bounds("minimo 2500") == (2500, None)


# --- Remainder stripping ----------------------------------------------------

def test_remainder_strip():
    c = extract_atk_constraint("dragons with more than 4000 ATK")
    assert c.remainder_query == "dragons with"

    c = extract_atk_constraint("dragões com mais de 4000 de ataque")
    assert c.remainder_query == "dragões"

    c = extract_atk_constraint("cards >= 2500 ATK")
    assert c.remainder_query == "cards"


def test_whitespace_collapse():
    c = extract_atk_constraint("dragons   with   more than   4000 ATK")
    assert c.min_atk == 4000
    assert c.remainder_query == "dragons with"


def test_empty_remainder_returns_original():
    c = extract_atk_constraint("more than 4000 ATK")
    assert (c.min_atk, c.max_atk) == (4000, None)
    assert c.remainder_query == "more than 4000 ATK"


def test_identity_passthrough_when_no_constraint():
    q = "blue eyes white dragon"
    assert extract_atk_constraint(q) == AtkConstraint(
        min_atk=None, max_atk=None, remainder_query=q
    )


def test_determinism():
    q = "dragons with more than 4000 ATK"
    assert extract_atk_constraint(q) == extract_atk_constraint(q)


# --- prepare_search_query (single entry point) ------------------------------

def test_prepare_search_query_plain():
    assert prepare_search_query("dragons with more than 4000 ATK", None, None) == (
        "dragons with",
        4000,
        None,
    )


def test_prepare_search_query_merges_explicit():
    assert prepare_search_query("dragons with more than 4000 ATK", 4500, None) == (
        "dragons with",
        4500,
        None,
    )
    assert prepare_search_query("dragons with more than 4000 ATK", None, 3000) == (
        "dragons with",
        4000,
        3000,
    )


# --- Fix 1: equality must defer to postfix suffixes --------------------------

def test_eq_defers_to_postfix_suffix():
    assert _bounds("monsters with 2500 ATK or more") == (2500, None)
    assert _bounds("cards with 2500 ATK or less") == (None, 2500)
    assert _bounds("monstros com 2500 de ataque ou mais") == (2500, None)
    # Suffix forms without an explicit keyword still work as before.
    assert _bounds("2500 or more") == (2500, None)
    assert _bounds("dragons with 2500 or more ATK") == (2500, None)


# --- Fix 2: "not less than" is a min form ------------------------------------

def test_not_less_than_is_min():
    assert _bounds("not less than 2500 ATK") == (2500, None)
    assert _bounds("no less than 2500") == (2500, None)


# --- Fix 3: bare ranges require the keyword trailer --------------------------

def test_bare_ranges_without_keyword_ignored():
    queries = [
        "level 8 and 7",
        "8 and 7",
        "cards with 3 and 5 effects",
        "level 8-12 monsters",
        "2 a 3 cards",
    ]
    for q in queries:
        assert _bounds(q) == (None, None), q


def test_bare_ranges_with_keyword_still_parse():
    queries = [
        "1000-2000 ATK",
        "1000 to 2000 ATK",
        "1000 a 2000 de ataque",
        "1000 and 2000 ATK",
    ]
    for q in queries:
        assert _bounds(q) == (1000, 2000), q


# --- Fix 4: vague quantifiers are not equality -------------------------------

def test_vague_quantifiers_not_equality():
    queries = [
        "up to 2000 ATK",
        "about 2000 ATK",
        "around 2000 ATK",
        "approximately 2000 ATK",
        "approx 2000 ATK",
        "~2000 ATK",
    ]
    for q in queries:
        assert _bounds(q) == (None, None), q


# --- Fix 5: verb-sense "attack" is not an equality keyword -------------------

def test_attack_verb_not_equality():
    assert _bounds("monsters that attack 2 times") == (None, None)
    assert _bounds("2500 attack power") == (2500, 2500)


# --- Fix 6: "ATK of 2500" equality -------------------------------------------

def test_atk_of_equality():
    assert _bounds("ATK of 2500") == (2500, 2500)
