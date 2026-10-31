"""The Docker files and workflows, checked without Docker: the serve image's HEALTHCHECK command, the paths
that trigger the docker workflow, and the smoke tests both workflows run before anything is published."""

import json
import os
import re
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import yaml

from predfoot.paths import REPO_ROOT

DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile.cpu"
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
SMOKE_SCRIPTS = ("docker/smoke-cli.sh", "docker/smoke-serve.sh")


def _stage(name: str) -> str:
    """The text of one build stage of the Dockerfile."""
    text = DOCKERFILE.read_text(encoding="utf-8")
    start = re.search(rf"^FROM \S+ AS {name}$", text, flags=re.MULTILINE)
    assert start is not None, name
    end = re.search(r"^FROM ", text[start.end() :], flags=re.MULTILINE)
    return text[start.start() : start.end() + end.start()] if end else text[start.start() :]


def _healthcheck_code() -> str:
    match = re.search(r"HEALTHCHECK [^\n]*\\\n\s*CMD (\[.*\])", _stage("serve"))
    assert match is not None
    argv = json.loads(match.group(1))
    assert argv[:2] == ["python", "-c"]
    return argv[2]


class _Health(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(self.server.status if self.path == "/healthz" else 404)
        self.end_headers()

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def health_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Health)
    server.status = 200
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def _dead_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _probe(code: str, port: int, proxy: str | None) -> int:
    env = {k: v for k, v in os.environ.items() if k.lower() not in ("no_proxy", "predfoot_port")}
    env["PREDFOOT_PORT"] = str(port)
    if proxy is not None:
        for key in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
            env[key] = proxy
    return subprocess.run([sys.executable, "-c", code], env=env, timeout=20, check=False).returncode


def test_the_serve_healthcheck_probes_predfoot_port_and_ignores_proxies(health_server):
    code = _healthcheck_code()
    port = health_server.server_address[1]
    dead_proxy = f"http://127.0.0.1:{_dead_port()}"
    assert _probe(code, port, proxy=None) == 0
    # The Docker CLI can hand the host's proxy variables to containers: the probe must not go through them.
    assert _probe(code, port, proxy=dead_proxy) == 0
    plain_urlopen = (
        f"import urllib.request; urllib.request.urlopen('http://127.0.0.1:{port}/healthz', timeout=4)"
    )
    assert _probe(plain_urlopen, port, proxy=dead_proxy) != 0  # what the probe did before
    health_server.status = 503
    assert _probe(code, port, proxy=None) != 0
    assert _probe(code, _dead_port(), proxy=None) != 0


def test_the_serve_image_listens_on_every_interface_by_default():
    serve = _stage("serve")
    assert re.search(r"^ENV PREDFOOT_HOST=0\.0\.0\.0", serve, flags=re.MULTILINE)
    assert "PREDFOOT_PORT=8000" in serve
    assert 'CMD ["serve"]' in serve  # options added to the command keep the host from the environment


def _covered(path: str, patterns: list[str]) -> bool:
    return any(
        path == p or (p.endswith("/**") and (path == p[:-3] or path.startswith(p[:-3] + "/")))
        for p in patterns
    )


def test_the_docker_workflow_runs_when_a_file_the_image_copies_changes():
    workflow = yaml.safe_load((WORKFLOWS / "docker.yml").read_text(encoding="utf-8"))
    triggers = workflow.get("on", workflow.get(True))  # YAML 1.1 reads the key `on` as True
    patterns = triggers["pull_request"]["paths"]
    sources = []
    for line in DOCKERFILE.read_text(encoding="utf-8").splitlines():
        words = line.split()
        if words[:1] == ["COPY"] and not any(w.startswith("--from=") for w in words):
            sources += [w.rstrip("/") for w in words[1:-1] if not w.startswith("--")]
    assert {"pyproject.toml", "uv.lock", "README.md", "LICENSE", "src", "ATTRIBUTION.md"} <= set(sources)
    missing = [s for s in [*sources, *SMOKE_SCRIPTS] if not _covered(s, patterns)]
    assert missing == [], f"the docker workflow does not run when these change: {missing}"


def _steps(workflow: str, job: str) -> list[dict]:
    return yaml.safe_load((WORKFLOWS / workflow).read_text(encoding="utf-8"))["jobs"][job]["steps"]


def _index(steps: list[dict], needle: str) -> int:
    found = [i for i, step in enumerate(steps) if needle in str(step.get("run", step.get("uses", "")))]
    assert len(found) == 1, (needle, found)
    return found[0]


def test_both_workflows_run_the_smoke_scripts():
    for script in SMOKE_SCRIPTS:
        assert os.access(REPO_ROOT / script, os.X_OK), f"{script} is not executable"
    _index(_steps("docker.yml", "cli"), "docker/smoke-cli.sh predfoot:cli")
    serve = _steps("docker.yml", "serve")
    assert _index(serve, "docker/smoke-cli.sh predfoot:serve") < _index(
        serve, "docker/smoke-serve.sh predfoot:serve"
    )


def test_the_release_pushes_nothing_before_the_smoke_tests():
    steps = _steps("release.yml", "publish")
    login = _index(steps, "docker/login-action")
    assert _index(steps, "docker/smoke-cli.sh predfoot:release") < login
    serve = _index(steps, "docker/smoke-serve.sh predfoot:release")
    assert serve < login
    assert steps[serve]["if"] == "matrix.target == 'serve'"
    pushes = [i for i, step in enumerate(steps) if step.get("with", {}).get("push") is True]
    assert pushes, "no push step"
    assert min(pushes) > login
