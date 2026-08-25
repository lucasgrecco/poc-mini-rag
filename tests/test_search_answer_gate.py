"""DB-free tests for the OPENAI_API_KEY gate around the generated answer.

Retrieval (embedding, structured filters, hybrid RRF, reranking) works without
an OpenAI key; only `get_model_answer` needs one. These tests stub out the
engine, the embedder, retrieval and the answer call, so nothing here touches a
database or the network.
"""

import pytest

import app.search as search


class _FakeConn:
    """Minimal stand-in for a SQLAlchemy connection used as a context manager."""

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class _FakeEngine:
    def connect(self):
        return _FakeConn()


RESULTS = [{"name": "Dragon Master Lords", "content": "ATK: 5000 | DEF: 5000"}]


@pytest.fixture
def answer_calls(monkeypatch):
    """Stub the whole pipeline; return the list of get_model_answer calls."""
    calls = []

    def _fake_answer(question, results_db):
        calls.append((question, results_db))
        return "generated answer"

    monkeypatch.setattr(search, "engine", _FakeEngine())
    monkeypatch.setattr(search, "get_embedding", lambda text: [0.0, 0.0])
    monkeypatch.setattr(search, "hybrid_search_sync", lambda *a, **k: list(RESULTS))
    monkeypatch.setattr(search, "rerank", lambda *a, **k: list(RESULTS))
    monkeypatch.setattr(search, "get_model_answer", _fake_answer)
    return calls


def test_no_key_skips_answer(monkeypatch, capsys, answer_calls):
    monkeypatch.setattr(search, "OPENAI_API_KEY", None)

    search.search_card("strong dragons")

    assert answer_calls == []
    out = capsys.readouterr().out
    # Retrieval output still prints, unchanged.
    assert "Top 1 Results:" in out
    assert "1. Dragon Master Lords" in out
    assert "ATK: 5000 | DEF: 5000" in out
    # The answer section is replaced by a one-line notice.
    assert "AI Answer" not in out
    assert "(no OPENAI_API_KEY set - retrieval only, skipping the generated answer)" in out


def test_key_present_attempts_answer(monkeypatch, capsys, answer_calls):
    monkeypatch.setattr(search, "OPENAI_API_KEY", "sk-test-not-a-real-key")

    search.search_card("strong dragons")

    assert answer_calls == [("strong dragons", RESULTS)]
    out = capsys.readouterr().out
    assert "Top 1 Results:" in out
    assert "AI Answer:\ngenerated answer" in out
    assert "no OPENAI_API_KEY set" not in out


def test_get_client_without_key_raises(monkeypatch):
    monkeypatch.setattr(search, "OPENAI_API_KEY", None)
    monkeypatch.setattr(search, "_client", None)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        search._get_client()


def test_get_client_is_cached(monkeypatch):
    monkeypatch.setattr(search, "OPENAI_API_KEY", "sk-test-not-a-real-key")
    monkeypatch.setattr(search, "_client", None)

    client = search._get_client()
    assert client is not None
    # Second call reuses the singleton instead of constructing a new client.
    assert search._get_client() is client
