# predfoot

**Where every player stands, read from a TV picture, and how far off it could be.**

predfoot reads the pitch markings in a football broadcast to work out where the camera is and where it
points, then places each player on the pitch in metres. Every position comes with an error zone that should
contain the true position 9 times in 10. predfoot also smooths the camera over time, flags frames whose camera
is probably wrong, and measures how camera errors inflate physical stats such as distance covered. The
players' boxes come from the dataset's annotators. predfoot does not detect players, so its results measure
the camera and the error zones.

A portfolio project by Ya-Sin, a data-science master's student.

<a href="https://huggingface.co/spaces/Ya-SinZ/predfoot"><img src="assets/showcase/hero.jpg" width="420" alt="predfoot on public match footage seen through a virtual TV camera: the pitch lines redrawn from the camera it recovered, and the same players on a pitch diagram with their error zones"></a>

### [Open the live page](https://huggingface.co/spaces/Ya-SinZ/predfoot)

*SoccerTrack v2 footage (CC BY 4.0) seen through a virtual TV camera, as an illustration; the players' feet
come from the dataset's boxes. The results below come from SoccerNet matches, whose images cannot be shown
for licence reasons.*

## The problem

Tracking data from TV broadcasts starts by working out where the camera is pointing, and any error there
shifts every player on screen at the same time. Frame after frame, these shifts inflate distance covered and
high-speed running, and the positions come with no sign of how far off they might be.

## Results on three unseen matches

One test, on 49 clips from three SoccerNet matches kept out of development. Each result compares predfoot with
the same pipeline minus the smoothing step: same network, same camera solver, each frame worked out on its own.

| | Without smoothing | With smoothing (predfoot) |
|---|---|---|
| **Accuracy**, distance within which 9 players in 10 are placed | 1.8 m | 1.2 m |
| **Physical stats**, extra distance covered from camera wobble | 26.1 % | 3.4 % |
| **Physical stats**, extra high-speed running | 72.4 % | 5.5 % |
| **Error zones**, true positions inside their 90 % zone | 90.7 % | 93.1 %, target missed |
| **Bad-camera alarm**, wrong camera estimates caught with 1 frame in 10 set aside | 34 % (best single warning sign) | 55 % |

- The error zones came out a little too large. Allowing for chance, 93.1 % is too far from the promised 90 %,
  so the target set before the test was missed. They are still 27 % smaller than a fixed circle making the
  same promise.
- The physical stats are measured 5 times a second against the dataset's smoothed tracks, which carry some
  noise of their own, so they show what camera wobble adds rather than absolute accuracy.
- The alarm was tested on the frame-by-frame estimates, where 1 frame in 10 is wrong. Beating the best single
  warning sign was the target set before the test (ranking score 0.91 against 0.80), and it was met.

**Tested once, on matches never used during development, under a protocol frozen in advance and locked with
checksums in git** ([preregistration](docs/preregistration.md)). Every result is published, including the
missed target ([test report](reports/test_run.md)). The live page ends with the study written up as a
one-page technical note.

## Smoothing the camera

![Two pitch diagrams of the same clip: without smoothing, the error zones jump and swell; with smoothing, they stay small and steady](assets/showcase/smoothing.gif)

*A clip from a SoccerNet match used during development, drawn on pitch diagrams because SoccerNet images
cannot be shown, and picked because it shows the effect clearly. Without smoothing (left), the zones jump and
swell whenever the camera estimate shakes. With smoothing (right) they stay small and steady, except for one
moment when both estimates lose the pitch ("camera lost").*

## From picture to pitch

1. **Find the pitch markings.** A public neural network, PnLCalib, used as published, finds the pitch lines
   and the points where they cross.
2. **Recover the camera.** The pitch is a known 105 × 68 m shape, so the markings give away where the camera
   is and where it points. This is calibration; predfoot also estimates how uncertain it is.
3. **Smooth over time.** A broadcast camera pans, tilts and zooms from a fixed spot. predfoot uses that to
   smooth the camera across the clip and to catch frames that do not fit.
4. **Place each player.** Each player's feet, taken from the dataset's boxes, are projected onto the pitch.
   The zone around them is tuned on past matches to contain the true position 9 times in 10, and an alarm
   flags frames whose camera looks wrong.

## Limits

- Tested on three matches from one league, filmed in one broadcast style.
- The players' boxes come from the dataset's annotators. Detecting and tracking players is outside the
  project.
- The error zones were slightly too large on the test matches.
- The physical stats were checked against the dataset's smoothed tracks, which carry some noise of their own.
- The alarm and the physical-stat checks run in the evaluation scripts, not yet in the command-line tool.

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

- [Technical overview](docs/technical.md): the method in detail, all results, reproduction, layout
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
diagrams. Pitch-marking network: PnLCalib weights by Gutiérrez-Pérez and Agudo, CC BY 4.0. The showcase page
uses D3, Motion and Lenis ([licences](assets/showcase/vendor/LICENSES.txt)). Every third-party resource is
credited in [ATTRIBUTION.md](ATTRIBUTION.md). predfoot is an independent project, not affiliated with
SkillCorner.
