# ShoulderLab

ShoulderLab is a research workspace for markerless shoulder range-of-motion
analysis. It focuses on converting reconstructed human motion into
shoulder-specific kinematic descriptors, including torso-compensated arm angles,
temporal movement features, reach-space summaries, and analysis reports for
functional shoulder assessment.

## Project Layout

- `shoulderlab/`: ShoulderLab Python package.
- `scripts/shoulderlab.py`: command line entrypoint.
- `third_party/HSMR/`: upstream HSMR submodule. Do not edit this directory for ShoulderLab changes.
- `third_party/Pi3/`: upstream Pi3 submodule for visual geometry experiments.
- `third_party/vggt/`: upstream VGGT submodule. Do not edit this directory for ShoulderLab changes.
- `data_inputs/`: local input videos, models, checkpoints, and body-model assets.
- `data_outputs/`: generated HSMR reconstructions and ShoulderLab analysis outputs.
- `docs/`: shoulder ROM notes and temporal analysis writeups.

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

Create and activate the ShoulderLab conda environment:

```bash
conda create -n shoulderlab python=3.10
conda activate shoulderlab
```

Install the CUDA-matched PyTorch wheel first, then install the project
requirements:

```bash
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
pip install --no-build-isolation -r requirements.txt
```

## Data Preparation

ShoulderLab uses the same HSMR model assets, but stores them at the ShoulderLab root:

```text
data_inputs/
├── UUCM/
├── shoulder/
│   └── subjectXX/
│       └── NNN_movement_name/
│           ├── cam_a.mp4
│           ├── cam_b.mp4
│           └── cam_c.mp4
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

Shoulder multiview videos are stored under `data_inputs/shoulder` as
`subjectXX/NNN_movement_name/cam_*.mp4`. In each movement trial, `cam_a`,
`cam_b`, and `cam_c` are synchronized left, right, and center views respectively.
See `docs/Shoulder_Data_Structure.md` for the full data layout convention, and
`docs/Shoulder_Multiview_Pi3_HSMR_Pipeline.md` for the planned Pi3/HSMR
multiview reconstruction and validation workflow.

For fresh setup, follow the download instructions in `third_party/HSMR/docs/SETUP.md`, but place files under ShoulderLab `data_inputs/`, not under `third_party/HSMR/data_inputs/`.

## Command Line

All commands go through one entrypoint:

```bash
python scripts/shoulderlab.py --help
```

ShoulderLab-owned progress messages use timestamped logs such as
`[ShoulderLab] YYYY-MM-DD HH:MM:SS INFO ...`.

Available commands:

```bash
python scripts/shoulderlab.py analyze --help
python scripts/shoulderlab.py analyze-uucm --help
python scripts/shoulderlab.py hsmr --help
python scripts/shoulderlab.py hsmr-uucm --help
python scripts/shoulderlab.py summary --help
```

## Quick Start

Analyze one existing HSMR `.npy` result:

```bash
python scripts/shoulderlab.py analyze \
  -i data_outputs/UUCM/HSMR-001_Flexion.npy \
  -o data_outputs/UUCM/analysis \
  -s right \
  --skip-video
```

Analyze all UUCM `.npy` files and write temporal summaries into the same `analysis` directory:

```bash
python scripts/shoulderlab.py analyze-uucm \
  --input-dir data_outputs/UUCM \
  --output-dir data_outputs/UUCM/analysis \
  --skip-video
```

Regenerate temporal feature summaries from existing result JSON files:

```bash
python scripts/shoulderlab.py summary data_outputs/UUCM/analysis
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

`summary` writes:

```text
temporal_feature_summary.csv
Temporal_Feature_Noise_Report.md
```

## Third-party Code

HSMR, Pi3, and VGGT are included as Git submodules:

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

Keep ShoulderLab changes in `shoulderlab/`, `scripts/`, or `docs/`. Do not patch `third_party/HSMR`, `third_party/Pi3`, or `third_party/vggt` for project-specific behavior.

## License

ShoulderLab is released under the MIT License. See `LICENSE`.

The upstream HSMR, Pi3, and VGGT submodules have their own licenses; see each upstream repository. External body models, checkpoints, datasets, Detectron2, SKEL, SMPL/SMPLify assets, Pi3/Pi3X weights, and other dependencies may have their own licenses or access terms. Follow those upstream terms when redistributing or publishing results.
