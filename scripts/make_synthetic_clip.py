"""Write the synthetic clip used by the tracer bullet, its tests and the Docker smoke test.

    uv run python scripts/make_synthetic_clip.py            # → tests/fixtures/synthetic_clip.json

A panning broadcast camera at (0, -45, 18) m (focal 1800 px, 1920 × 1080, 25 fps, 3 s), 22 players moving
smoothly, calibration failing on 10 frames and a spectator detected above the horizon. See
``predfoot.pipeline.synthetic.make_broadcast_clip``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from predfoot.paths import REPO_ROOT
from predfoot.pipeline.synthetic import dumps_clip, make_broadcast_clip

DEFAULT_OUT = REPO_ROOT / "tests" / "fixtures" / "synthetic_clip.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="where to write the clip JSON")
    args = parser.parse_args(argv)
    clip, _ = make_broadcast_clip()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(dumps_clip(clip), encoding="utf-8")
    n_feet = sum(len(f["feet"]) for f in clip["frames"])
    print(f"wrote {args.out} ({len(clip['frames'])} frames, {n_feet} foot points)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
