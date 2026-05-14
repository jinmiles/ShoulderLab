# ShoulderLab

ShoulderLab is a shoulder range-of-motion analysis workspace built on top of the upstream HSMR project. The upstream HSMR repository is kept as a clean Git submodule under `third_party/HSMR`; ShoulderLab code, datasets, and outputs live outside that directory.

## Project Layout

- `shoulderlab/`: ShoulderLab Python package.
- `scripts/shoulderlab.py`: command line entrypoint.
- `third_party/HSMR/`: upstream HSMR submodule. Do not edit this directory for ShoulderLab changes.
- `data_inputs/`: local input videos, models, checkpoints, and body-model assets.
- `data_outputs/`: generated HSMR reconstructions and ShoulderLab analysis outputs.
- `docs/`: shoulder ROM notes and Q2 analysis writeups.

## Setup

Clone this repository with submodules:

```bash
git clone --recurse-submodules <SHOULDERLAB_REPO_URL>
cd ShoulderLab
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

Use the existing HSMR conda environment. HSMR and the project dependencies are
already installed there, so do not run `pip install` as a routine setup step:

```bash
conda activate hsmr
python --version  # expected: Python 3.8.x
```

If the `hsmr` environment does not exist, create it once:

```bash
conda create -n hsmr python=3.8
conda activate hsmr
```

The default ShoulderLab environment is the `hsmr` conda env with Python 3.8, matching HSMR's version-pinned setup notes in `third_party/HSMR/docs/requirements_py3.8.txt`. HSMR also reports testing on Python 3.10, but use the existing Python 3.8 `hsmr` env first unless you have a reason to change it. Match your `torch` build to your local CUDA setup.

## Data Preparation

ShoulderLab uses the same HSMR model assets, but stores them at the ShoulderLab root:

```text
data_inputs/
├── UUCM/
├── backbone/
│   └── vitpose_backbone.pth
├── body_models/
│   ├── J_regressor_SKEL_mix_MALE.pkl
│   ├── J_regressor_SMPL_MALE.pkl
│   ├── SMPL_to_J19.pkl
│   └── skel/
└── released_models/
    └── HSMR-ViTH-r1d1/
```

Minimum required assets:

- HSMR checkpoint: `data_inputs/released_models/HSMR-ViTH-r1d1`
- SKEL model: `data_inputs/body_models/skel`
- auxiliary regressors: `data_inputs/body_models/*.pkl`
- ViTPose backbone checkpoint: `data_inputs/backbone/vitpose_backbone.pth`
- local videos: `data_inputs/UUCM/*.MP4`

For fresh setup, follow the download instructions in `third_party/HSMR/docs/SETUP.md`, but place files under ShoulderLab `data_inputs/`, not under `third_party/HSMR/data_inputs/`.

## Command Line

All commands go through one entrypoint:

```bash
python scripts/shoulderlab.py --help
```

Available commands:

```bash
python scripts/shoulderlab.py analyze --help
python scripts/shoulderlab.py analyze-uucm --help
python scripts/shoulderlab.py hsmr --help
python scripts/shoulderlab.py hsmr-uucm --help
python scripts/shoulderlab.py q2-summary --help
```

## Quick Start

Analyze one existing HSMR `.npy` result:

```bash
python scripts/shoulderlab.py analyze \
  -i data_outputs/UUCM/HSMR-001_Flexion.npy \
  -o data_outputs/UUCM/q2_analysis \
  -s right \
  --skip-video
```

Analyze all UUCM `.npy` files:

```bash
python scripts/shoulderlab.py analyze-uucm \
  --input-dir data_outputs/UUCM \
  --output-dir data_outputs/UUCM/analysis \
  --skip-video
```

Generate Q2 temporal feature summaries:

```bash
python scripts/shoulderlab.py q2-summary data_outputs/UUCM/q2_analysis
```

## HSMR Reconstruction

Run upstream HSMR on one video or image folder using ShoulderLab-owned input/output paths:

```bash
python scripts/shoulderlab.py hsmr \
  -i data_inputs/UUCM/001_Flexion.MP4 \
  -o data_outputs/UUCM
```

Run HSMR reconstruction for every UUCM video:

```bash
python scripts/shoulderlab.py hsmr-uucm \
  --input-dir data_inputs/UUCM \
  --output-dir data_outputs/UUCM
```

For videos, HSMR mesh rendering can be slow. Use `--ignore-skel`, reduce `--max-instances`, or lower `--mesh-bs` if runtime or GPU memory is a problem.

## Outputs

HSMR reconstruction writes files such as:

```text
data_outputs/UUCM/HSMR-001_Flexion.npy
data_outputs/UUCM/HSMR-001_Flexion.mp4
```

ShoulderLab ROM analysis writes:

```text
*_results.json
*_angles.png
*_temporal_features.png
*_reach_3d.png
*_combined_skeleton_reach.png
```

`q2-summary` writes:

```text
q2_temporal_feature_summary.csv
Q2_Temporal_Feature_Noise_Report.md
```

## Third-party Code

HSMR is included as a Git submodule:

```bash
git submodule status
```

To update HSMR later:

```bash
cd third_party/HSMR
git fetch origin
git checkout <desired_commit_or_tag>
cd ../..
git add third_party/HSMR
```

Keep ShoulderLab changes in `shoulderlab/`, `scripts/`, or `docs/`. Do not patch `third_party/HSMR` for project-specific behavior.

## License

ShoulderLab is released under the MIT License. See `LICENSE`.

The upstream HSMR submodule is also MIT licensed; see `third_party/HSMR/LICENSE`. External body models, checkpoints, datasets, Detectron2, SKEL, SMPL/SMPLify assets, and other dependencies may have their own licenses or access terms. Follow those upstream terms when redistributing or publishing results.
