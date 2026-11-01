# predfoot

**Where every player stands, read from a TV picture, and how far off it could be.**

predfoot takes a frame of a football broadcast, works out from the pitch markings where the camera stands and
where it points, and places every player on the pitch in metres. Each position comes with an error zone: an
area built to hold the player's true spot 9 times out of 10. predfoot also warns when the camera estimate is
probably wrong, smooths the camera over time, and measures how camera errors distort physical stats such as
distance covered. It does not detect players: it starts from the player boxes drawn by the dataset's
annotators, so its results measure the camera and the error zones, not detection.

A portfolio project by Ya-Sin, a data-science master's student.

<a href="https://huggingface.co/spaces/Ya-SinZ/predfoot"><img src="assets/showcase/hero.jpg" width="420" alt="predfoot on public match footage seen through a virtual TV camera: the pitch lines redrawn from the camera it recovered, and the same players on a pitch diagram with their error zones"></a>

### [Open the live page: predfoot in 30 seconds](https://huggingface.co/spaces/Ya-SinZ/predfoot)

*Public footage (SoccerTrack v2, CC BY 4.0) seen through a virtual TV camera, illustrating the output; the
players' feet come from the dataset's boxes. The results below come from SoccerNet matches, whose images
cannot be shown for licence reasons.*

## Why it matters

Much of football's tracking data now comes from ordinary TV broadcasts. To turn a picture into positions,
software first has to work out where the camera is pointing, and when that estimate wobbles, every player on
screen moves with it. The wobble inflates stats such as distance covered and high-speed running, and the
positions rarely say how far off they might be.

## Results

predfoot was tested once, on three SoccerNet matches that played no part in its development. It is compared
with itself without the smoothing step: the same network and the same camera solver, with each frame worked
out on its own.

| | On the three test matches |
|---|---|
| **Accuracy** | 9 players in 10 placed within 1.2 m of their labelled spot, against 1.8 m without smoothing |
| **Physical stats** | Camera wobble adds 3.4 % to the distance players cover, against 26.1 % without smoothing, and 5.5 % to high-speed running, against 72.4 %. Measured against the dataset's smoothed tracks, which are not exact either (the raw labelled positions give 17 % more distance and 40 % more high-speed running than they do): this is the distance wobble adds, not absolute accuracy |
| **Error zones** | The true position fell inside its 90 % zone 93.1 % of the time. The target set before the test was 90 %, within the margin of error, so this is a **missed target**, on the cautious side: the zones were a bit too large (without smoothing they landed at 90.7 %). They are still 27 % smaller than a one-size-fits-all circle making the same promise |
| **Bad-camera alarm** | Setting aside the 10 % of frames it finds most suspicious catches 55 % of the frames whose camera estimate is wrong (more than 10 px from the labelled pitch lines). It was tested on the frame-by-frame estimates, where 1 frame in 10 is wrong, and, as the test required, it ranks those frames better than the best single warning signal (0.91 against 0.80) |

**Tested once, on matches never used during development, under a protocol frozen in advance and locked with
checksums in git** ([preregistration](docs/preregistration.md)). Every result is published, including the
missed target ([test report](reports/test_run.md)).

## Smoothing the camera

![Two pitch diagrams of the same clip: frame by frame, the error zones jump and swell; smoothed over time, they stay small and steady](assets/showcase/smoothing.gif)

*A clip from a SoccerNet match used during development, drawn on a pitch diagram because SoccerNet images
cannot be shown for licence reasons, and picked to show the effect. Frame by frame (left), the error zones jump
and swell whenever the camera estimate wobbles; smoothed over time (right), they stay small and steady, apart
from one moment when both lose the pitch ("camera lost").*

## How it works

1. **Find the pitch markings.** An existing public neural network (PnLCalib, used as published, not
   retrained) spots the lines of the pitch, and the points where they cross, in each frame.
2. **Recover the camera.** The pitch is a known 105 × 68 m shape, so those markings reveal where the camera is
   and where it points. This is called calibration; predfoot also measures how sure it can be of it.
3. **Smooth over time.** A TV camera pans, tilts and zooms around an almost fixed point. predfoot uses this to
   smooth the camera across the clip and to catch frames that do not fit.
4. **Place each player, with an error zone.** Each player's feet, taken from the dataset's boxes, are projected
   onto the pitch with a zone built to hold the true spot 9 times in 10, tuned on past matches so it keeps that
   promise. An alarm flags frames whose camera estimate looks wrong.

## Limits

- Tested on three matches from one league and one broadcast style.
- Players come from boxes drawn by the dataset's annotators: detecting and tracking them is not part of the
  project.
- The error zones were slightly cautious on the test matches.
- The physical stats were measured against the dataset's own smoothed tracks, which are not exact either: they
  show what camera wobble adds, not absolute accuracy.
- The bad-camera alarm and the physical-stat measurements run in the evaluation scripts, not yet in the
  command-line tool.

## For engineers

The method, every number and how to reproduce it are in the [technical overview](docs/technical.md). Outputs
follow the SkillCorner open-data format at 10 fps (kloppy loads them), with the error zones as extra fields.
A quick start that needs no dataset, on synthetic detections of a known camera:

```bash
uv sync
uv run predfoot run tests/fixtures/golden/clip.json --out out-golden/ \
    --detections tests/fixtures/golden/detections.json --conformal tests/fixtures/golden/conformal.json
uv run predfoot qa out-golden/    # rebuilds the run's one-page HTML report
```

- [Technical overview](docs/technical.md): how it works in detail, all results, reproduction, layout
- [Reports](reports/README.md): every generated report, with the command and commit behind it
- [Preregistration](docs/preregistration.md) and [test report](reports/test_run.md)
- [HTTP API and latency](docs/serving.md), [Docker images](docs/docker.md)
- [Showcase page](site/page.html), built for the live page by `scripts/build_site.py`

> Preregistration v1, frozen on 2026-11-01 by commit `8640b9a` at 11:56 UTC, about an hour before the test run
> started (12:58 UTC), on code commit `a46c15aaaea4`: `docs/preregistration.md` SHA-256 `25bf24bf31da3027…`,
> artifacts digest `20f06142cba1522c…`, `docs/preregistration.lock.json` SHA-256 `191bb06dbb75aba8…`; hashes in
> [`docs/preregistration.lock.json`](docs/preregistration.lock.json). The repository was private when the
> protocol was frozen: the timing can be checked in its git history.

## Licence and credits

Apache-2.0. Footage: SoccerTrack v2, Scott et al., CC BY 4.0, reprojected to a virtual camera, cropped,
downscaled and overlaid. Test data: SoccerNet SN-GSR (Somers et al.), shown only as numbers, plots and pitch
diagrams. Pitch-marking network: PnLCalib weights by Gutiérrez-Pérez and Agudo, CC BY 4.0. Every third-party
resource is credited in [ATTRIBUTION.md](ATTRIBUTION.md). predfoot is an independent project, not affiliated
with SkillCorner.
