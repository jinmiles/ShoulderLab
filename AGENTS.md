# AGENTS.md

Instructions for AI coding agents working in this repository.

## Mission

ShoulderLab is a project-owned shoulder range-of-motion analysis layer on top of
upstream computer-vision and body-model projects. Keep ShoulderLab code,
documents, data paths, and analysis behavior under project control while
preserving `third_party/HSMR` and `third_party/vggt` as clean upstream
dependencies.

## Ground Rules

- Treat this file as the highest-priority repo guidance after direct user
  instructions.
- Read the relevant code before editing. Prefer small, targeted changes that
  match the existing style.
- Do not modify `third_party/HSMR/`, `third_party/HSMR/thirdparty/SKEL/`, or
  `third_party/vggt/` for ShoulderLab-specific behavior unless the user
  explicitly asks for an upstream patch. Put integration code in `shoulderlab/`,
  `scripts/`, or `docs/`.
- Do not commit generated data, local model assets, videos, checkpoints, plots,
  or analysis outputs. `data_inputs/`, `data_outputs/`, and `.cache/` are local
  workspace directories.
- When the user asks for tests, or when you run tests/experiments on your own,
  create a dedicated folder for that test under `tests/`, write any ad hoc test
  code/scripts inside that folder, and write outputs/artifacts there as well.
  Run those test scripts from the appropriate conda environment unless the user
  explicitly asks for a different runtime. `tests/` is ignored by Git and should
  be preserved until the user deletes it manually or explicitly agrees that you
  may delete it.
- Preserve existing user changes. Check `git status --short` before editing and
  avoid reverting unrelated files.
- Keep generated outputs deterministic where practical. Avoid hidden global state
  except for explicit HSMR path configuration in `shoulderlab.paths`.

## Git And Commits

- Do not create commits without user confirmation.
- When a meaningful unit of work is complete, it is acceptable to suggest a
  commit. Ask the user before committing.
- Before committing, review `git status --short` and include only the intended
  changes. Do not stage unrelated user work.
- Write commit messages in English.
- Use the Google Developers Blockly commit message convention:
  `https://developers.google.com/blockly/guides/contribute/get-started/commits`.
- Commit message format:

  ```text
  <type>: <description>

  [optional body]

  [optional footer(s)]
  ```

- Use lowercase commit types. Allowed types:
  - `chore`: routine or automated maintenance tasks.
  - `deprecate`: deprecating functionality.
  - `feat`: adding new functionality.
  - `fix`: fixing bugs or errors.
  - `release`: release-related changes.
- Mark breaking changes by appending `!` after the type, for example
  `feat!: change analysis result schema`.
- Keep the description non-empty, concise, and under 256 characters.
- If a body or footer is needed, separate it from the description with a blank
  line and keep each line under 256 characters.
- Before committing, run the smallest relevant validation command when feasible
  and mention any validation that could not be run.

## Repository Map

- `shoulderlab/`: Project Python package.
  - `cli.py`: command dispatch and public CLI arguments.
  - `paths.py`: repository paths and HSMR import/path integration.
  - `hsmr.py`: wrapper around upstream HSMR demo execution.
  - `analyze.py`: HSMR output loading, SKEL joint recovery, ROM analysis flow.
  - `rom.py`: local coordinate system, shoulder angles, temporal features, and
    visualization helpers.
  - `summary.py`: CSV and Markdown summary generation for temporal features.
- `scripts/shoulderlab.py`: thin executable wrapper for the package CLI.
- `docs/`: research notes, math notes, reports, and project writeups.
- `third_party/HSMR/`: upstream HSMR Git submodule. Keep clean.
- `third_party/vggt/`: upstream VGGT Git submodule. Keep clean.
- `data_inputs/`: local videos, body models, checkpoints, and other assets.
- `data_outputs/`: generated reconstructions, JSON, CSV, plots, and videos.

## Environment

Use the existing `hsmr` conda environment unless the user says otherwise. HSMR
and its project dependencies are already installed in that environment; do not
run `pip install` as a routine setup step.

```bash
conda activate hsmr
python --version
```

The expected Python version is 3.8.x because the project follows HSMR's
Python 3.8 setup. Match `torch` and CUDA to the local machine.

## Common Commands

```bash
python scripts/shoulderlab.py --help
python scripts/shoulderlab.py analyze --help
python scripts/shoulderlab.py hsmr --help
python scripts/shoulderlab.py summary --help
```

Analyze one existing HSMR `.npy` result without rendering video:

```bash
python scripts/shoulderlab.py analyze \
  -i data_outputs/UUCM/HSMR-001_Flexion.npy \
  -o data_outputs/UUCM/analysis \
  -s right \
  --skip-video
```

Generate the temporal summary from existing result JSON files:

```bash
python scripts/shoulderlab.py summary data_outputs/UUCM/analysis
```

`analyze-uucm` should write per-sample plots/JSON and temporal summary CSV/Markdown
into the same `data_outputs/UUCM/analysis` directory. Do not reintroduce a
separate `analysis` default path.

## Validation

There is currently no dedicated test suite in this repository. Use the smallest
validation that fits the change:

```bash
python -m py_compile shoulderlab/*.py scripts/shoulderlab.py
```

For CLI changes:

```bash
python scripts/shoulderlab.py --help
python scripts/shoulderlab.py analyze --help
```

For analysis or feature changes, run on a small existing local `.npy` file with
`--skip-video` when available. Do not require GPU-heavy HSMR reconstruction
unless the change specifically touches reconstruction behavior.

## Coding Style

- Use straightforward Python with type hints where they clarify public function
  inputs and outputs.
- Keep path handling based on `pathlib.Path`.
- Use structured APIs for JSON, CSV, NumPy, and Path operations. Avoid ad hoc
  string parsing when a standard parser is available.
- Keep comments short and useful. Explain non-obvious math, coordinate-system
  choices, or HSMR integration details.
- Prefer project-level wrappers over patching upstream HSMR internals.
- Keep CLI defaults aligned with `README.md` and `shoulderlab.paths`.
- Keep user-facing output concise, but preserve existing progress logs when they
  help long-running GPU workflows.

## Data And Safety

- Local body models, checkpoints, videos, and subject data may have licensing or
  privacy constraints. Do not move them into tracked files.
- Do not print large arrays, full JSON payloads, model weights, or personal data
  in logs or docs.
- When adding examples, use placeholder paths or paths already documented in the
  README. Do not invent claims about data quality or clinical validity.
- Shoulder ROM metrics are research/analysis outputs, not medical diagnoses.
  Keep wording precise in docs and reports.

## Change Checklist

Before finishing:

1. Confirm `git status --short` only shows intended file changes, plus any
   pre-existing user changes.
2. Run a syntax check or a narrower relevant command when feasible.
3. Update `README.md` or `docs/` if CLI behavior, outputs, setup, or analysis
   semantics changed.
4. Mention any validation that could not be run because data, GPU, or external
   dependencies were unavailable.
