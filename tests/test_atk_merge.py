"""DB-free tests for ATK bound merging and the SQL filter builder."""

import logging

from app.query_parser import merge_atk_bounds, prepare_search_query
from app.retrieval import build_filter_clause


def test_extracted_only():
    assert merge_atk_bounds(None, None, 4000, None) == (4000, None)
    assert merge_atk_bounds(None, None, None, 3000) == (None, 3000)


def test_explicit_only():
    assert merge_atk_bounds(4500, None, None, None) == (4500, None)
    assert merge_atk_bounds(None, 2500, None, None) == (None, 2500)


def test_none_none():
    assert merge_atk_bounds(None, None, None, None) == (None, None)


def test_min_takes_max():
    # Extracted 4000 + explicit 4500 -> 4500 (tighter lower bound wins).
    assert merge_atk_bounds(4500, None, 4000, None) == (4500, None)
    # Extracted 4000 + explicit 3000 -> 4000 (tighter lower bound wins).
    assert merge_atk_bounds(3000, None, 4000, None) == (4000, None)


def test_max_takes_min():
    # Extracted 3000 + explicit 2500 -> 2500 (tighter upper bound wins).
    assert merge_atk_bounds(None, 2500, None, 3000) == (None, 2500)
    # Extracted 3000 + explicit 4000 -> 3000 (tighter upper bound wins).
    assert merge_atk_bounds(None, 4000, None, 3000) == (None, 3000)


def test_mixed():
    # Extracted min 4000 + explicit max 4500 -> (4000, 4500).
    assert merge_atk_bounds(None, 4500, 4000, None) == (4000, 4500)


def test_build_filter_clause_combined():
    fragment, params = build_filter_clause(min_atk=4000, max_atk=4500)
    assert "atk >= :min_atk" in fragment
    assert "atk <= :max_atk" in fragment
    assert " AND " in fragment
    assert params == {"min_atk": 4000, "max_atk": 4500}


def test_build_filter_clause_empty():
    assert build_filter_clause() == ("", {})


def test_prepare_search_query_intersection_passthrough():
    # Explicit min + no extracted constraint: the explicit bound passes through.
    assert prepare_search_query("blue eyes white dragon", 4500, None) == (
        "blue eyes white dragon",
        4500,
        None,
    )


def test_merge_four_way_combination():
    # Explicit (4000, 5000) AND extracted (4500, 3000) -> (4500, 3000).
    assert merge_atk_bounds(4000, 5000, 4500, 3000) == (4500, 3000)


def test_contradictory_bounds_log_warning(caplog):
    # Explicit min 4000 + extracted "less than 2000" -> (4000, 2000) is
    # contradictory (min > max) and must emit a warning.
    with caplog.at_level(logging.WARNING, logger="app.query_parser"):
        result = prepare_search_query("cards with less than 2000 ATK", 4000, None)
    assert result == ("cards with", 4000, 2000)
    assert any("contradict" in record.getMessage().lower() for record in caplog.records)
