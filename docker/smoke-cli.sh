#!/usr/bin/env bash
# Smoke test of a predfoot CPU image, run from the repository root: docker/smoke-cli.sh IMAGE
#
# The checks the docker workflow runs on every pull request and the release workflow runs before pushing:
# the image runs as uid 10001; predfoot run on the synthetic clip, then validate; the QA report holds no URL
# and no script, and predfoot qa rebuilds it byte for byte; the real stages reproduce the golden outputs.
# It runs on both targets (the serve image holds the command-line tools too).
#
# Environment: SMOKE_DIR, where outputs go (default smoke/, git-ignored); EXPECTED_VERSION and
# EXPECTED_GIT_SHA, when set, must match `predfoot version` and the git_sha of meta.json.
set -euo pipefail

image="${1:?usage: docker/smoke-cli.sh IMAGE}"
work="$(mkdir -p "${SMOKE_DIR:-smoke}" && cd "${SMOKE_DIR:-smoke}" && pwd)"
out="$work/out"
golden="$work/golden-out"

group() { echo "::group::$*"; }  # folded sections in the GitHub Actions log
endgroup() { echo "::endgroup::"; }
fail() {
  echo "FAILED: $*" >&2
  exit 1
}

group "The image runs as the unprivileged user"
uid="$(docker run --rm --entrypoint id "$image" -u)"  # an assignment, so that a failing docker run stops here
[ "$uid" = 10001 ] || fail "the image runs as uid '$uid', expected 10001"
if [ -n "${EXPECTED_VERSION:-}" ]; then
  version="$(docker run --rm "$image" version)"
  [ "$version" = "$EXPECTED_VERSION" ] ||
    fail "predfoot version prints '$version', expected '$EXPECTED_VERSION' (src/predfoot/__init__.py and pyproject.toml must agree)"
fi
endgroup

group "predfoot run on the synthetic clip, then predfoot validate"
rm -rf "$out"
mkdir -p "$out"
chmod 0777 "$out"  # the container user (uid 10001) writes here
docker run --rm -v "$PWD/tests/fixtures:/in:ro" -v "$out:/out" "$image" \
  run /in/synthetic_clip.json --out /out --fps 10
docker run --rm -v "$out:/out:ro" "$image" validate /out/tracking.jsonl --match /out/match.json
[ -s "$out/minimap.png" ] || fail "no minimap.png"
jq -e '.frames.exported == 30 and .frames.abstained == 4' "$out/metrics.json" >/dev/null ||
  fail "metrics.json: expected 30 export frames, 4 abstained"
if [ -n "${EXPECTED_GIT_SHA:-}" ]; then
  jq -e --arg sha "$EXPECTED_GIT_SHA" '.git_sha == $sha' "$out/meta.json" >/dev/null ||
    fail "meta.json does not record commit $EXPECTED_GIT_SHA"
fi
endgroup

group "The QA report is self-contained and predfoot qa rebuilds it"
[ -s "$out/qa_report.html" ] || fail "no qa_report.html"
if grep -Eiq 'https?://|<script' "$out/qa_report.html"; then
  fail "qa_report.html references a URL or holds a script"
fi
cp "$out/qa_report.html" "$work/first-report.html"
docker run --rm -v "$out:/out" "$image" qa /out
cmp "$out/qa_report.html" "$work/first-report.html" || fail "predfoot qa wrote other bytes than predfoot run"
endgroup

group "The real stages reproduce the golden outputs"
rm -rf "$golden"
mkdir -p "$golden"
chmod 0777 "$golden"
docker run --rm -v "$PWD/tests/fixtures/golden:/golden:ro" -v "$golden:/out" "$image" \
  run /golden/clip.json --detections /golden/detections.json --conformal /golden/conformal.json --out /out
docker run --rm --entrypoint python -v "$PWD/tests/fixtures/golden:/golden:ro" -v "$golden:/out:ro" \
  "$image" -m predfoot.io.compare /out /golden/expected
endgroup

echo "cli smoke test passed: $image"
