"""Run upstream HSMR through ShoulderLab path configuration."""

from __future__ import annotations

import sys
from typing import Dict, Tuple
from pathlib import Path

from shoulderlab.log import get_logger
from shoulderlab.paths import DATA_INPUTS, DATA_OUTPUTS, DEFAULT_MODEL_ROOT, configure_hsmr_paths


logger = get_logger()


def prepare_mesh_with_small_batches(pipeline, pd_params, batch_size: int = 200) -> Tuple[Dict, Dict]:
    """Upstream HSMR mesh preparation with a lower SKEL batch size to avoid OOM."""
    configure_hsmr_paths()

    import torch
    from lib.platform.sliding_batches import asb

    v_skin_all, v_skel_all = [], []
    for bw in asb(total=len(pd_params["poses"]), bs_scope=batch_size, enable_tqdm=True):
        skel_outputs = pipeline.skel_model(
            poses=pd_params["poses"][bw.sid:bw.eid].to(pipeline.device),
            betas=pd_params["betas"][bw.sid:bw.eid].to(pipeline.device),
        )
        v_skin_all.append(skel_outputs.skin_verts.detach().cpu())
        v_skel_all.append(skel_outputs.skel_verts.detach().cpu())

    v_skel_all = torch.cat(v_skel_all, dim=0)
    v_skin_all = torch.cat(v_skin_all, dim=0)
    return {
        "v": v_skin_all,
        "f": pipeline.skel_model.skin_f,
    }, {
        "v": v_skel_all,
        "f": pipeline.skel_model.skel_f,
    }


def run_hsmr(
    input_path: Path,
    output_path: Path = DATA_OUTPUTS / "demos",
    model_root: Path = DEFAULT_MODEL_ROOT,
    device: str = "cuda:0",
    det_bs: int = 10,
    det_mis: int = 512,
    rec_bs: int = 300,
    mesh_bs: int = 200,
    max_instances: int = 5,
    ignore_skel: bool = False,
    have_caption: bool = False,
) -> None:
    """Run upstream HSMR while keeping inputs and outputs under ShoulderLab."""
    logger.info("Running HSMR on %s", input_path)
    configure_hsmr_paths()
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    import exp.run_demo as hsmr_run_demo
    import lib.kits.hsmr_demo as hsmr_demo

    def _prepare_mesh(pipeline, pd_params):
        return prepare_mesh_with_small_batches(pipeline, pd_params, batch_size=mesh_bs)

    hsmr_demo.prepare_mesh = _prepare_mesh
    hsmr_run_demo.prepare_mesh = _prepare_mesh

    argv = [
        "shoulderlab hsmr",
        "--input_path",
        str(input_path),
        "--output_path",
        str(output_path),
        "--model_root",
        str(model_root),
        "--device",
        device,
        "--det_bs",
        str(det_bs),
        "--det_mis",
        str(det_mis),
        "--rec_bs",
        str(rec_bs),
        "--max_instances",
        str(max_instances),
    ]
    if ignore_skel:
        argv.append("--ignore_skel")
    if have_caption:
        argv.append("--have_caption")

    old_argv = sys.argv
    try:
        sys.argv = argv
        hsmr_run_demo.main()
    finally:
        sys.argv = old_argv
    logger.info("HSMR output directory: %s", output_path.resolve())


def run_uucm_hsmr(
    input_dir: Path = DATA_INPUTS / "UUCM",
    output_dir: Path = DATA_OUTPUTS / "UUCM",
    **hsmr_kwargs,
) -> None:
    """Run HSMR reconstruction for all UUCM videos."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(list(input_dir.glob("*.mp4")) + list(input_dir.glob("*.MP4")))
    if not videos:
        raise SystemExit(f"No videos found in {input_dir}")

    for video in videos:
        logger.info("Reconstructing %s", video.name)
        run_hsmr(
            input_path=video,
            output_path=output_dir,
            **hsmr_kwargs,
        )
