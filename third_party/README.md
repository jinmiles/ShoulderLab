# Third-party Repositories

This directory is for clean upstream repositories used by ShoulderLab.

- `HSMR/`: Git submodule for `https://github.com/IsshikiHugh/HSMR.git`

Do not edit files inside third-party repositories for ShoulderLab changes. Put project code in `shoulderlab/` or `scripts/`, and keep inputs/outputs under the repository-level `data_inputs/` and `data_outputs/` directories.

Initialize submodules with:

```bash
git submodule update --init --recursive
```

Use the repository-level `README.md` for environment setup. The expected environment is the existing `conda` env named `hsmr` with Python 3.8.
