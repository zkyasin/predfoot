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


def test_serve_loads_a_conformal_model(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    uvicorn = pytest.importorskip("uvicorn")
    from fastapi.testclient import TestClient

    from tests.unit.serve_fakes import mondrian_model

    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: calls.append(app))
    model = tmp_path / "conformal.json"
    model.write_text(mondrian_model().to_json())
    result = CliRunner().invoke(app, ["serve", "--conformal", str(model)])
    assert result.exit_code == 0, result.output
    assert TestClient(calls[0]).get("/healthz").json()["models"]["conformal"].startswith("mondrian@")
    model.write_text('{"alpha": 0.2, "edges": {}, "quantiles": {}, "counts": {}, "pooled": 9.0}')
    result = CliRunner().invoke(app, ["serve", "--conformal", str(model)])
    assert result.exit_code == 2
    assert "90 % regions" in result.output


def test_data_dir_follows_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("PREDFOOT_DATA_DIR", str(tmp_path))
    assert paths.data_dir() == tmp_path.resolve()
    assert paths.index_dir() == tmp_path.resolve() / "index"


def test_data_dir_defaults_to_the_repo(monkeypatch):
    monkeypatch.delenv("PREDFOOT_DATA_DIR", raising=False)
    assert paths.data_dir() == paths.REPO_ROOT / "data"
