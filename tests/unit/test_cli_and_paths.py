import tomllib

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


def test_serve_takes_host_and_port_from_the_environment(monkeypatch):
    """The serve image sets PREDFOOT_HOST=0.0.0.0, so `serve --conformal ...` listens on every interface."""
    pytest.importorskip("fastapi")
    uvicorn = pytest.importorskip("uvicorn")
    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: calls.append(kwargs))
    env = {"PREDFOOT_HOST": "0.0.0.0", "PREDFOOT_PORT": "8124"}
    assert CliRunner(env=env).invoke(app, ["serve"]).exit_code == 0
    assert CliRunner(env=env).invoke(app, ["serve", "--host", "127.0.0.1"]).exit_code == 0
    assert calls == [{"host": "0.0.0.0", "port": 8124}, {"host": "127.0.0.1", "port": 8124}]


def test_the_package_version_is_the_pyproject_version():
    """`predfoot version`, /healthz and meta.json print __version__; the release tag is checked against
    pyproject.toml. Bump both together."""
    pyproject = tomllib.loads((paths.REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert __version__ == pyproject["project"]["version"]


def test_serve_checks_the_device():
    result = CliRunner().invoke(app, ["serve", "--device", "gpu"])
    assert result.exit_code == 2
    assert "auto, cpu, mps, cuda" in result.output


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
