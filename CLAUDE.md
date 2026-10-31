# Working rules

Guidance for anyone (human or AI assistant) changing this repository.

## Commands

```bash
uv sync                                   # core + dev; add --extra vision for torch
uv run pytest -m "not slow and not data and not network and not vision"   # what CI runs
uv run pytest                             # everything available locally
uv run ruff check . && uv run ruff format --check . && uv run mypy
uv run python scripts/check_licenses.py
```

Run these before every commit; CI runs the same steps.

## Conventions (ADR-0001)

- **One internal pitch frame, SkillCorner's**: origin at the centre spot, metres, x towards the right-hand goal
  of the main camera, y towards the far touchline. SoccerNet and SoccerTrack v2 have y the other way round:
  convert only in `predfoot/io/` with `predfoot.io.conventions`, covariances included.
- Pixels refer to the original frame, OpenCV convention. Time is the native frame index internally; exports
  are at 10 fps.
- Shared types are in `predfoot/contracts.py`. Do not redefine them in modules.
- Randomness: seed from `(clip_id, frame)`; tests use the `rng` fixture.

## Code

- Python 3.12, type hints everywhere in `src/`, English names, docstrings and comments.
- Geometry and statistics are written test-first: finite-difference Jacobians, round trips on synthetic
  cameras, OpenCV as an oracle, coverage simulations with fixed seeds.
- Mark tests that need datasets `@pytest.mark.data`, internet `network`, torch `vision`, long runs `slow`.
- Experiments write to `runs/<date>_<git-sha>/` (config, metrics, figures). Reports with aggregate numbers
  go to `reports/`.

## Data and licences (ADR-0002)

- Never commit datasets, frames, labels or weights. Downloads go to `PREDFOOT_DATA_DIR`.
- No SoccerNet pixels in anything public; demo pixels come from SoccerTrack v2 (CC BY 4.0) only.
- No GPL, AGPL or non-commercial dependencies (`scripts/check_licenses.py`). Do not copy code from PnLCalib's
  GPL utilities, sn-gamestate, TrackLab or BroadTrack; the MIT HRNet model file is the only exception, kept
  with its header.

## Git

- Never push to `main`. One branch per ticket group (`m1/geometry`, `m0/data`…); the owner opens the PRs.
- Conventional commit messages (`feat:`, `fix:`, `test:`, `docs:`, `chore:`).
