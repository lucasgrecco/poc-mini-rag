"""Tests for the CLI search card pipeline and keyless import safety.

Covers the two local-mode mechanisms in app/search.py:
- import safety: app.search must import without OPENAI_API_KEY (lazy chat client)
- provider gate: the AI answer is only generated/printed when
  EMBEDDING_PROVIDER == "openai"; local mode prints results only.

All tests are DB-free and env-free: the client/DB are mocked and the
provider is controlled via monkeypatch on the app.search module globals
(config binds EMBEDDING_PROVIDER at import time, so os.environ is not used).
"""

import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import app.search as search

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS = [
    {"name": "Blue-Eyes White Dragon", "content": "A legendary dragon."},
    {"name": "Dark Magician", "content": "The ultimate wizard."},
]


def _run_search_card(monkeypatch, capsys, provider, model_answer=None):
    """Run search_card with all external dependencies mocked."""
    monkeypatch.setattr(search, "EMBEDDING_PROVIDER", provider)
    monkeypatch.setattr(
        search, "EMBEDDING_COLUMN", "embedding" if provider == "openai" else "embedding_local"
    )

    engine = mock.MagicMock()
    monkeypatch.setattr(search, "engine", engine)
    monkeypatch.setattr(search, "get_embedding", mock.Mock(return_value=[0.1, 0.2]))
    monkeypatch.setattr(search, "hybrid_search_sync", mock.Mock(return_value=RESULTS))
    monkeypatch.setattr(search, "rerank", mock.Mock(return_value=RESULTS))

    get_model_answer = mock.Mock(return_value=model_answer)
    monkeypatch.setattr(search, "get_model_answer", get_model_answer)

    search.search_card("blue eyes white dragon")

    return get_model_answer, capsys.readouterr().out


def test_local_mode_skips_model_answer(monkeypatch, capsys):
    """Local mode prints results but never calls or prints the AI answer."""
    get_model_answer, out = _run_search_card(monkeypatch, capsys, "local")

    assert "🏆 Top 2 Results:" in out
    assert "Blue-Eyes White Dragon" in out
    assert "Dark Magician" in out
    assert "💡 AI Answer" not in out
    get_model_answer.assert_not_called()


def test_openai_mode_calls_and_prints_model_answer(monkeypatch, capsys):
    """OpenAI mode still calls get_model_answer and prints its output."""
    answer = "Blue-Eyes White Dragon is a legendary dragon."
    get_model_answer, out = _run_search_card(
        monkeypatch, capsys, "openai", model_answer=answer
    )

    get_model_answer.assert_called_once_with("blue eyes white dragon", RESULTS)
    assert "💡 AI Answer:" in out
    assert answer in out


def test_import_app_search_without_key():
    """app.search imports cleanly when OPENAI_API_KEY is absent."""
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    result = subprocess.run(
        [sys.executable, "-c", "import app.search"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


def test_import_app_search_with_empty_key():
    """app.search imports cleanly when OPENAI_API_KEY is empty."""
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = ""
    result = subprocess.run(
        [sys.executable, "-c", "import app.search"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
