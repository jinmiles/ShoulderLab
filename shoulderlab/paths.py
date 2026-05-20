"""Project paths and HSMR integration helpers."""

from __future__ import annotations

import sys
import os
from pathlib import Path


SHOULDERLAB_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_ROOT = SHOULDERLAB_ROOT / "third_party"
HSMR_ROOT = THIRD_PARTY_ROOT / "HSMR"
PI3_ROOT = THIRD_PARTY_ROOT / "Pi3"

DATA_INPUTS = SHOULDERLAB_ROOT / "data_inputs"
DATA_OUTPUTS = SHOULDERLAB_ROOT / "data_outputs"
DEFAULT_MODEL_ROOT = DATA_INPUTS / "released_models" / "HSMR-ViTH-r1d1"

MPLCONFIGDIR = SHOULDERLAB_ROOT / ".cache" / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))


def configure_hsmr_paths() -> None:
    """Make HSMR importable while keeping HSMR inputs/outputs outside third_party."""
    sys.dont_write_bytecode = True
    hsmr_root = str(HSMR_ROOT)
    if hsmr_root not in sys.path:
        sys.path.insert(0, hsmr_root)

    from lib.platform.proj_manager import ProjManager as PM

    PM.root = HSMR_ROOT
    PM.configs = HSMR_ROOT / "configs"
    PM.inputs = DATA_INPUTS
    PM.outputs = DATA_OUTPUTS


def configure_pi3_paths() -> None:
    """Make the upstream Pi3 package importable without patching third_party."""
    pi3_root = str(PI3_ROOT)
    if pi3_root not in sys.path:
        sys.path.insert(0, pi3_root)
