# Docker images and release

`docker/Dockerfile.cpu` builds two CPU images without torch, both running as an unprivileged user
(uid 10001):

| Target | Contents | Default command |
|---|---|---|
| `cli` (default) | The command-line tools | `predfoot --help` |
| `serve` | The tools and the HTTP API (extra `serve`); `EXPOSE 8000`, a `HEALTHCHECK` on `/healthz` | `predfoot serve`, on `PREDFOOT_HOST=0.0.0.0` and `PREDFOOT_PORT=8000` |

```bash
docker build -f docker/Dockerfile.cpu --build-arg GIT_SHA="$(git rev-parse HEAD)" -t predfoot:cli .
mkdir -p out && chmod 0777 out
docker run --rm -v "$PWD/tests/fixtures:/in:ro" -v "$PWD/out:/out" \
    predfoot:cli run /in/synthetic_clip.json --out /out
docker run --rm -v "$PWD/out:/out:ro" predfoot:cli validate /out/tracking.jsonl --match /out/match.json

docker build -f docker/Dockerfile.cpu --target serve -t predfoot:serve .
docker run --rm -p 8000:8000 predfoot:serve
```

In the `serve` image, `/healthz`, `/v1/localize` and `/metrics` work; `/v1/calibrate` answers 503 because the
keypoint networks need torch (extra `vision`), which the CPU images leave out. `predfoot serve` reads its
interface and port from `PREDFOOT_HOST` and `PREDFOOT_PORT` when the command line does not set them, so
`docker run -p 8000:8000 predfoot:serve serve --conformal /models/conformal.json` still listens on every
interface; the `HEALTHCHECK` probes `PREDFOOT_PORT`, without proxies.

Two scripts smoke-test the images, locally or in CI: `docker/smoke-cli.sh IMAGE` (non-root user,
`predfoot run`, `validate` and `qa`, then the real stages on the golden fixture, whose outputs must match
`tests/fixtures/golden/expected/`; it runs on both targets) and `docker/smoke-serve.sh IMAGE` (a healthy
`HEALTHCHECK`, `/healthz`, `POST /v1/localize` with 200 and finite values, `POST /v1/calibrate` with 503,
`/metrics`). The `docker` workflow builds both targets on every pull request that touches what the images
contain and runs them; nothing is pushed there.

## Release

Bump the version in `pyproject.toml` and in `src/predfoot/__init__.py` (a unit test checks that they agree),
merge, then push a tag `v<version>`. The `release` workflow checks that the tag matches the package version,
runs the smoke scripts of the `docker` workflow on each image (and `predfoot version`), the HTTP checks
included for the `serve` image, and only then pushes them to the GitHub Container Registry with
`GITHUB_TOKEN`, with OCI labels (source, revision, version, licence):

| Image | Tags |
|---|---|
| `ghcr.io/zkyasin/predfoot` (cli) | `<version>`, `sha-<commit>` |
| `ghcr.io/zkyasin/predfoot` (serve) | `<version>-serve`, `sha-<commit>-serve` |
