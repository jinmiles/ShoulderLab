# Shoulder Input Data Structure

This document describes the local shoulder multiview video layout under
`data_inputs/shoulder`. The raw videos are local workspace data and should not be
committed.

## Directory Layout

```text
data_inputs/shoulder/
`-- subjectXX/
    `-- NNN_movement_name/
        |-- cam_a.mp4
        |-- cam_b.mp4
        `-- cam_c.mp4
```

- `subjectXX`: anonymized subject identifier, for example `subject01`.
- `NNN_movement_name`: movement trial identifier and shoulder motion name.
- `cam_a.mp4`, `cam_b.mp4`, `cam_c.mp4`: synchronized multiview videos for the
  same subject and movement trial.

## Camera Convention

The three videos in a trial are time-synchronized and represent different camera
views of the same movement:

| File | View |
|:--|:--|
| `cam_a.mp4` | left view |
| `cam_b.mp4` | right view |
| `cam_c.mp4` | center view |

Code that reads a movement trial can assume the three camera files are already
synchronized at the frame/time level. If a future dataset includes an
unsynchronized trial, store synchronization metadata explicitly rather than
overloading these filenames.

## Current Local Dataset

Observed in the local workspace on 2026-05-20:

- Subjects: `subject01` through `subject06`.
- Standard movement trials:
  - `001_flexion`
  - `002_abduction`
  - `003_internal_rotation`
  - `004_external_rotation`
  - `005_internal_rotation_abduction`
  - `006_external_rotation_abduction`
  - `007_circumduction`
- Additional trial currently present for `subject02`:
  - `008_back`

Most subject and movement folders contain all three camera files. At the time of
writing, `subject06` does not include `007_circumduction`.

## Processing Notes

- Keep outputs outside `data_inputs/shoulder`; use `data_outputs/` for HSMR
  reconstructions, analysis JSON, plots, summaries, and derived videos.
- Preserve the `subjectXX/NNN_movement_name/cam_*.mp4` layout so batch scripts can
  discover subject, movement, and camera view from paths.
- Treat the videos as research data, not clinical diagnostic data.
