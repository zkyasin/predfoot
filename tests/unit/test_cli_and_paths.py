import pytest
from typer.testing import CliRunner

from predfoot import __version__, paths
from predfoot.cli import app


def test_version_command():
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_serve_command_runs_the_api(monkeypatch):
    pytest.importorskip("fastapi")
    uvicorn = pytest.importorskip("uvicorn")
    from fastapi import FastAPI

    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: calls.append((app, kwargs)))
    result = CliRunner().invoke(app, ["serve", "--host", "0.0.0.0", "--port", "8123"])
    assert result.exit_code == 0, result.output
    [(served, kwargs)] = calls
    assert isinstance(served, FastAPI)
    assert kwargs == {"host": "0.0.0.0", "port": 8123}


def test_data_dir_follows_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("PREDFOOT_DATA_DIR", str(tmp_path))
    assert paths.data_dir() == tmp_path.resolve()
    assert paths.index_dir() == tmp_path.resolve() / "index"


def test_data_dir_defaults_to_the_repo(monkeypatch):
    monkeypatch.delenv("PREDFOOT_DATA_DIR", raising=False)
    assert paths.data_dir() == paths.REPO_ROOT / "data"
