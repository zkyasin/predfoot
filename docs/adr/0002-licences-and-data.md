# ADR-0002: Licences and data handling

- Status: accepted
- Date: 2026-10-31

## Decision

**Code.** The repository is Apache-2.0 and written clean-room:
- no GPL, AGPL or non-commercial dependencies; `scripts/check_licenses.py` enforces this in CI. LGPL and
  unknown licences need a reviewed exception; MPL-2.0 is accepted because its copyleft only covers files we
  would modify, and we use those packages unmodified;
- no code copied from PnLCalib's GPL utilities, No-Bells-Just-Whistles, sn-gamestate, TrackLab or BroadTrack;
- the only third-party source files are the MIT HRNet model definition, kept with its header and licence text
  and credited, and the showcase page's vendored D3, Motion and Lenis builds (`assets/showcase/vendor/`, with
  their licences);
- facts needed for interoperability (for example which keypoint index a pretrained model assigns to which
  pitch intersection) may be taken from public documentation or code; the coordinates themselves are recomputed
  from FIFA dimensions in our own code.

**Weights.** PnLCalib's CC BY 4.0 weights are downloaded at runtime and credited. They are never committed.

**Data.**
- Nothing from any dataset is committed: no frames, labels or derived per-frame tables. Reports with aggregate
  numbers are fine.
- SoccerNet frames never appear in anything public (README, video, demo). SoccerNet results are shown as
  metrics, plots and minimaps.
- Public pixels come only from SoccerTrack v2 (CC BY 4.0, consent covers redistribution), with attribution and
  a note that frames are cropped, downscaled and overlaid.
- IDSSE (CC BY 4.0) derived data and plots are fine with the citation.
- A tiny excerpt of SkillCorner open data (MIT) may be used as a test fixture, with credit.

**Evaluation hygiene.** SN-GSR test games stayed locked until the preregistration was frozen, then were
evaluated once by `scripts/run_test_once.py`; development scripts still refuse them. The official SoccerNet
evaluator is only ever run outside the repository.
