#!/usr/bin/env bash
# Smoke test of the predfoot serve image, run from the repository root: docker/smoke-serve.sh IMAGE
#
# The checks the docker workflow runs on every pull request and the release workflow runs before pushing the
# serve image: started with its default command, the container's HEALTHCHECK turns healthy; GET /healthz;
# POST /v1/localize answers 200 with finite values (the committed request ends with a foot above the
# horizon, which must come back without a position); POST /v1/calibrate answers 503 (no torch in the CPU
# images); GET /metrics counts the requests. The server log is printed and the container removed at the end,
# pass or fail.
#
# Environment: SMOKE_DIR, where responses go (default smoke/, git-ignored); SMOKE_PORT, the host port
# published to the container's port 8000 (default 8000).
set -euo pipefail

image="${1:?usage: docker/smoke-serve.sh IMAGE}"
port="${SMOKE_PORT:-8000}"
work="$(mkdir -p "${SMOKE_DIR:-smoke}/serve" && cd "${SMOKE_DIR:-smoke}/serve" && pwd)"
url="http://127.0.0.1:$port"
name="predfoot-smoke-serve-$$"

group() { echo "::group::$*"; }  # folded sections in the GitHub Actions log
endgroup() { echo "::endgroup::"; }
fail() {
  echo "FAILED: $*" >&2
  exit 1
}
cleanup() {
  local status=$?
  echo "::group::Server log"
  docker logs "$name" 2>&1 || true
  echo "::endgroup::"
  docker rm -f "$name" >/dev/null 2>&1 || true
  exit "$status"
}

group "The image runs as the unprivileged user"
uid="$(docker run --rm --entrypoint id "$image" -u)"  # an assignment, so that a failing docker run stops here
[ "$uid" = 10001 ] || fail "the image runs as uid '$uid', expected 10001"
endgroup

group "The server starts with the default command and its HEALTHCHECK turns healthy"
docker run -d --name "$name" -p "$port:8000" "$image" >/dev/null
trap cleanup EXIT
health=starting
for _ in $(seq 1 45); do
  health="$(docker inspect --format '{{.State.Health.Status}}' "$name")"
  if [ "$health" = healthy ]; then
    break
  fi
  if [ "$(docker inspect --format '{{.State.Running}}' "$name")" != true ]; then
    fail "the container stopped (health: $health)"
  fi
  sleep 2
done
[ "$health" = healthy ] || fail "health status after 90 s: $health"
endgroup

group "GET /healthz"
curl -sf "$url/healthz" -o "$work/healthz.json" || fail "GET /healthz failed"
cat "$work/healthz.json"
echo
jq -e '.status == "ok" and .models == {}' "$work/healthz.json" >/dev/null ||
  fail "/healthz: expected status ok and no loaded model"
endgroup

group "POST /v1/localize answers 200 with finite values"
code="$(curl -s -o "$work/localize.json" -w '%{http_code}' -H 'content-type: application/json' \
  --data @tests/fixtures/golden/localize_request.json "$url/v1/localize")"
cat "$work/localize.json"
echo
[ "$code" = 200 ] || fail "POST /v1/localize answered $code"
jq -e '
  [.positions[] | select(.x != null)] as $located
  | ($located | length) == ((.positions | length) - 1)
  and .positions[-1].x == null
  and ([$located[] | .x, .y, .p90_radius_m, .quantile, (.pos_cov_m2 | flatten | .[])]
       | all(type == "number" and (isinfinite | not) and (isnan | not)))
' "$work/localize.json" >/dev/null ||
  fail "/v1/localize: expected finite positions for every foot but the last, above the horizon"
endgroup

group "POST /v1/calibrate answers 503 (no torch in this image)"
docker run --rm --entrypoint python "$image" -c \
  "import io, sys; from PIL import Image; b = io.BytesIO(); Image.new('RGB', (64, 36)).save(b, 'PNG'); sys.stdout.buffer.write(b.getvalue())" \
  >"$work/frame.png"
code="$(curl -s -o "$work/calibrate.json" -w '%{http_code}' -F "image=@$work/frame.png;type=image/png" \
  "$url/v1/calibrate")"
cat "$work/calibrate.json"
echo
[ "$code" = 503 ] || fail "POST /v1/calibrate answered $code, expected 503"
jq -e '.detail | test("torch")' "$work/calibrate.json" >/dev/null || fail "/v1/calibrate: the 503 does not name torch"
endgroup

group "GET /metrics counts the requests"
curl -sf -o "$work/metrics.txt" "$url/metrics" || fail "GET /metrics failed"
grep -q '^predfoot_http_requests_total' "$work/metrics.txt" || fail "/metrics has no predfoot_http_requests_total"
endgroup

echo "serve smoke test passed: $image"
