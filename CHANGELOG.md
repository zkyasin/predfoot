# Changelog

All notable changes are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/) and
the project uses [semantic versioning](https://semver.org/).

## [Unreleased]

### Added
- Project skeleton: uv, pre-commit, CI, licence check, package layout, coordinate conventions, data contracts,
  ADR-0001 (conventions) and ADR-0002 (licences and data).
- SkillCorner-format export: record writer and reader with the uncertainty extension fields, image corners
  clipped like SkillCorner's, minimal `match.json` that kloppy loads, 25 → 10 fps export by interpolation.
- Frame JSON Schema 1.0 and semantic validator; `predfoot validate` (also for genuine SkillCorner files).
- Contract tests against kloppy and a three-record excerpt of SkillCorner open data (MIT).
- Minimap with 90 % ellipses and greyed-out abstentions.
- `predfoot run`: stub tracer bullet processed in resumable blocks (byte-identical after a kill).
- CPU Docker image and a workflow that runs the tracer bullet in it.
