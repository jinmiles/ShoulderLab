"""Command line interface for ShoulderLab."""

from __future__ import annotations

import argparse
from pathlib import Path

from shoulderlab.log import configure_logging
from shoulderlab.paths import DATA_INPUTS, DATA_OUTPUTS, DEFAULT_MODEL_ROOT, SHOULDERLAB_ROOT


def _add_analysis_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-m", "--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("-d", "--device", default="cuda:0")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--sg-window-sec", type=float, default=0.33)
    parser.add_argument("--sg-polyorder", type=int, default=3)
    parser.add_argument("--peak-prominence-deg", type=float, default=None)
    parser.add_argument("--skel-bs", type=int, default=200)
    parser.add_argument("--skip-video", action="store_true")


def _analysis_kwargs(args: argparse.Namespace) -> dict:
    return {
        "model_root": args.model_root,
        "device": args.device,
        "fps": args.fps,
        "sg_window_sec": args.sg_window_sec,
        "sg_polyorder": args.sg_polyorder,
        "peak_prominence_deg": args.peak_prominence_deg,
        "skel_bs": args.skel_bs,
        "skip_video": args.skip_video,
    }


def _add_hsmr_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-m", "--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("-d", "--device", default="cuda:0")
    parser.add_argument("--det-bs", type=int, default=10)
    parser.add_argument("--det-mis", type=int, default=512)
    parser.add_argument("--rec-bs", type=int, default=300)
    parser.add_argument("--mesh-bs", type=int, default=200)
    parser.add_argument("--max-instances", type=int, default=5)
    parser.add_argument("--ignore-skel", action="store_true")
    parser.add_argument("--have-caption", action="store_true")


def _hsmr_kwargs(args: argparse.Namespace) -> dict:
    return {
        "model_root": args.model_root,
        "device": args.device,
        "det_bs": args.det_bs,
        "det_mis": args.det_mis,
        "rec_bs": args.rec_bs,
        "mesh_bs": args.mesh_bs,
        "max_instances": args.max_instances,
        "ignore_skel": args.ignore_skel,
        "have_caption": args.have_caption,
    }


def _add_shoulder_trial_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--subject", required=True)
    parser.add_argument("--movement", required=True)
    parser.add_argument("--reference-view", choices=["cam_a", "cam_b", "cam_c"], default="cam_c")
    parser.add_argument("--manifest-path", type=Path, default=DATA_OUTPUTS / "shoulder" / "manifests" / "shoulder_manifest.json")
    parser.add_argument("--input-root", type=Path, default=DATA_INPUTS / "shoulder")


def _add_shoulder_dataset_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--subject", default=None, help="Optional subject filter.")
    parser.add_argument("--movement", default=None, help="Optional movement filter.")
    parser.add_argument("--reference-view", choices=["cam_a", "cam_b", "cam_c"], default="cam_c")
    parser.add_argument("--manifest-path", type=Path, default=DATA_OUTPUTS / "shoulder" / "manifests" / "shoulder_manifest.json")
    parser.add_argument("--input-root", type=Path, default=DATA_INPUTS / "shoulder")


def _add_pi3_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pi3-model", choices=["pi3", "pi3x"], default="pi3")
    parser.add_argument("--pi3-ckpt", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=30)
    parser.add_argument("--sample-interval", type=int, default=None)
    parser.add_argument("--pixel-limit", type=int, default=255000)
    parser.add_argument("--max-pointcloud-points", type=int, default=200000)
    parser.add_argument("-d", "--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")


def _pi3_kwargs(args: argparse.Namespace) -> dict:
    return {
        "model_name": args.pi3_model,
        "checkpoint": args.pi3_ckpt,
        "device": args.device,
        "max_samples": args.max_samples,
        "sample_interval": args.sample_interval,
        "pixel_limit": args.pixel_limit,
        "max_pointcloud_points": args.max_pointcloud_points,
        "force": args.force,
    }


def _add_joints_fusion_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-m", "--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("-d", "--device", default="cuda:0")
    parser.add_argument("--skel-bs", type=int, default=200)
    parser.add_argument("--alignment-variant", choices=["static", "dynamic"], default="static")
    parser.add_argument("-s", "--side", choices=["right", "left", "both"], default="both")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--require-pi3", action="store_true")
    parser.add_argument("--force-joints", action="store_true")
    parser.add_argument("--force-fusion", action="store_true")


def _fusion_kwargs(args: argparse.Namespace) -> dict:
    return {
        "model_root": args.model_root,
        "device": args.device,
        "skel_bs": args.skel_bs,
        "alignment_variant": args.alignment_variant,
        "side": args.side,
        "fps": args.fps,
        "require_pi3": args.require_pi3,
        "force_joints": args.force_joints,
        "force_fusion": args.force_fusion,
    }


def _add_fused_analysis_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-s", "--side", choices=["right", "left", "both"], default="both")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--sg-window-sec", type=float, default=0.33)
    parser.add_argument("--sg-polyorder", type=int, default=3)
    parser.add_argument("--peak-prominence-deg", type=float, default=None)
    parser.add_argument("--skip-video", action="store_true")


def _fused_analysis_kwargs(args: argparse.Namespace) -> dict:
    return {
        "side": args.side,
        "fps": args.fps,
        "sg_window_sec": args.sg_window_sec,
        "sg_polyorder": args.sg_polyorder,
        "peak_prominence_deg": args.peak_prominence_deg,
        "skip_video": args.skip_video,
    }


def _add_shoulder_pipeline_options(parser: argparse.ArgumentParser) -> None:
    _add_pi3_options(parser)
    parser.add_argument("-m", "--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--det-bs", type=int, default=10)
    parser.add_argument("--det-mis", type=int, default=512)
    parser.add_argument("--rec-bs", type=int, default=300)
    parser.add_argument("--mesh-bs", type=int, default=200)
    parser.add_argument("--max-instances", type=int, default=5)
    parser.add_argument("--ignore-skel", action="store_true")
    parser.add_argument("--have-caption", action="store_true")
    parser.add_argument("--skel-bs", type=int, default=200)
    parser.add_argument("--alignment-variant", choices=["static", "dynamic"], default="static")
    parser.add_argument("-s", "--side", choices=["right", "left", "both"], default="both")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--sg-window-sec", type=float, default=0.33)
    parser.add_argument("--sg-polyorder", type=int, default=3)
    parser.add_argument("--peak-prominence-deg", type=float, default=None)
    parser.add_argument("--skip-pi3", action="store_true")
    parser.add_argument("--skip-hsmr", action="store_true")
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--require-pi3", action="store_true")


def _shoulder_pipeline_kwargs(args: argparse.Namespace) -> dict:
    pi3_kwargs = _pi3_kwargs(args)
    force = pi3_kwargs.pop("force")
    hsmr_kwargs = {
        "model_root": args.model_root,
        "device": args.device,
        "det_bs": args.det_bs,
        "det_mis": args.det_mis,
        "rec_bs": args.rec_bs,
        "mesh_bs": args.mesh_bs,
        "max_instances": args.max_instances,
        "ignore_skel": args.ignore_skel,
        "have_caption": args.have_caption,
    }
    fuse_kwargs = {
        "model_root": args.model_root,
        "device": args.device,
        "skel_bs": args.skel_bs,
        "alignment_variant": args.alignment_variant,
        "side": args.side,
        "fps": args.fps,
        "require_pi3": args.require_pi3,
    }
    analysis_kwargs = {
        "side": args.side,
        "fps": args.fps,
        "sg_window_sec": args.sg_window_sec,
        "sg_polyorder": args.sg_polyorder,
        "peak_prominence_deg": args.peak_prominence_deg,
        "skip_video": args.skip_video,
    }
    return {
        "skip_pi3": args.skip_pi3,
        "skip_hsmr": args.skip_hsmr,
        "force": force,
        "pi3_kwargs": pi3_kwargs,
        "hsmr_kwargs": hsmr_kwargs,
        "fuse_kwargs": fuse_kwargs,
        "analysis_kwargs": analysis_kwargs,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shoulderlab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Analyze one HSMR .npy/.npz output.")
    analyze.add_argument("-i", "--input-path", type=Path, required=True)
    analyze.add_argument("-o", "--output-dir", type=Path, default=DATA_OUTPUTS / "shoulder_analysis")
    analyze.add_argument("-s", "--side", choices=["right", "left", "both"], default="both")
    _add_analysis_options(analyze)

    analyze_uucm = subparsers.add_parser("analyze-uucm", help="Analyze all UUCM HSMR .npy outputs and write summaries.")
    analyze_uucm.add_argument("--input-dir", type=Path, default=DATA_OUTPUTS / "UUCM")
    analyze_uucm.add_argument("--output-dir", type=Path, default=DATA_OUTPUTS / "UUCM" / "analysis")
    analyze_uucm.add_argument("-s", "--side", choices=["right", "left", "both"], default="both")
    _add_analysis_options(analyze_uucm)

    hsmr = subparsers.add_parser("hsmr", help="Run HSMR on one video or image folder.")
    hsmr.add_argument("-i", "--input-path", type=Path, required=True)
    hsmr.add_argument("-o", "--output-dir", type=Path, default=DATA_OUTPUTS / "demos")
    _add_hsmr_options(hsmr)

    hsmr_uucm = subparsers.add_parser("hsmr-uucm", help="Run HSMR reconstruction for UUCM videos.")
    hsmr_uucm.add_argument("--input-dir", type=Path, default=DATA_INPUTS / "UUCM")
    hsmr_uucm.add_argument("--output-dir", type=Path, default=DATA_OUTPUTS / "UUCM")
    _add_hsmr_options(hsmr_uucm)

    summary = subparsers.add_parser("summary", help="Summarize temporal feature JSON outputs.")
    summary.add_argument("input_dir", type=Path, nargs="?", default=DATA_OUTPUTS / "UUCM" / "analysis")
    summary.add_argument("--docs-path", type=Path, default=SHOULDERLAB_ROOT / "docs" / "Temporal_Feature_Noise_Report.md")

    shoulder_manifest = subparsers.add_parser("shoulder-manifest", help="Discover and validate shoulder multiview trials.")
    shoulder_manifest.add_argument("--input-root", type=Path, default=DATA_INPUTS / "shoulder")
    shoulder_manifest.add_argument("--output-path", type=Path, default=DATA_OUTPUTS / "shoulder" / "manifests" / "shoulder_manifest.json")
    shoulder_manifest.add_argument("--subject", default=None)
    shoulder_manifest.add_argument("--movement", default=None)
    shoulder_manifest.add_argument("--reference-view", choices=["cam_a", "cam_b", "cam_c"], default="cam_c")
    shoulder_manifest.add_argument("--strict", action="store_true")

    pi3_shoulder = subparsers.add_parser("pi3-shoulder", help="Estimate Pi3/Pi3X camera poses for one shoulder trial.")
    _add_shoulder_trial_options(pi3_shoulder)
    _add_pi3_options(pi3_shoulder)

    hsmr_shoulder = subparsers.add_parser("hsmr-shoulder", help="Run HSMR for all camera views in one shoulder trial.")
    _add_shoulder_trial_options(hsmr_shoulder)
    _add_hsmr_options(hsmr_shoulder)
    hsmr_shoulder.add_argument("--force", action="store_true")

    fuse_shoulder = subparsers.add_parser("fuse-shoulder", help="Recover joints, align views, evaluate, and fuse one shoulder trial.")
    _add_shoulder_trial_options(fuse_shoulder)
    _add_joints_fusion_options(fuse_shoulder)

    analyze_shoulder = subparsers.add_parser("analyze-shoulder", help="Run ROM analysis on fused shoulder joints.")
    _add_shoulder_trial_options(analyze_shoulder)
    _add_fused_analysis_options(analyze_shoulder)

    shoulder_pipeline = subparsers.add_parser("shoulder-pipeline", help="Run manifest, Pi3, HSMR, fusion, and fused ROM analysis.")
    _add_shoulder_trial_options(shoulder_pipeline)
    _add_shoulder_pipeline_options(shoulder_pipeline)

    shoulder_dataset = subparsers.add_parser("shoulder-dataset", help="Run the shoulder pipeline for every valid trial.")
    _add_shoulder_dataset_options(shoulder_dataset)
    _add_shoulder_pipeline_options(shoulder_dataset)
    shoulder_dataset.add_argument("--continue-on-error", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logger = configure_logging()
    logger.info("Starting command: %s", args.command)

    try:
        if args.command == "analyze":
            from shoulderlab.analyze import run_analysis

            run_analysis(
                input_path=args.input_path,
                output_path=args.output_dir,
                side=args.side,
                **_analysis_kwargs(args),
            )
        elif args.command == "analyze-uucm":
            from shoulderlab.analyze import run_batch_analysis

            run_batch_analysis(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                side=args.side,
                **_analysis_kwargs(args),
            )
        elif args.command == "hsmr":
            from shoulderlab.hsmr import run_hsmr

            run_hsmr(
                input_path=args.input_path,
                output_path=args.output_dir,
                **_hsmr_kwargs(args),
            )
        elif args.command == "hsmr-uucm":
            from shoulderlab.hsmr import run_uucm_hsmr

            run_uucm_hsmr(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                **_hsmr_kwargs(args),
            )
        elif args.command == "summary":
            from shoulderlab.summary import summarize_analysis

            summarize_analysis(input_dir=args.input_dir, docs_path=args.docs_path)
        elif args.command == "shoulder-manifest":
            from shoulderlab.shoulder_pipeline import run_shoulder_manifest

            run_shoulder_manifest(
                input_root=args.input_root,
                output_path=args.output_path,
                subject=args.subject,
                movement=args.movement,
                reference_view=args.reference_view,
                strict=args.strict,
            )
        elif args.command == "pi3-shoulder":
            from shoulderlab.shoulder_pipeline import run_pi3_shoulder

            run_pi3_shoulder(
                subject=args.subject,
                movement=args.movement,
                manifest_path=args.manifest_path,
                input_root=args.input_root,
                reference_view=args.reference_view,
                **_pi3_kwargs(args),
            )
        elif args.command == "hsmr-shoulder":
            from shoulderlab.shoulder_pipeline import run_hsmr_shoulder

            run_hsmr_shoulder(
                subject=args.subject,
                movement=args.movement,
                manifest_path=args.manifest_path,
                input_root=args.input_root,
                reference_view=args.reference_view,
                force=args.force,
                **_hsmr_kwargs(args),
            )
        elif args.command == "fuse-shoulder":
            from shoulderlab.shoulder_pipeline import run_fuse_shoulder

            run_fuse_shoulder(
                subject=args.subject,
                movement=args.movement,
                manifest_path=args.manifest_path,
                input_root=args.input_root,
                reference_view=args.reference_view,
                **_fusion_kwargs(args),
            )
        elif args.command == "analyze-shoulder":
            from shoulderlab.shoulder_pipeline import run_analyze_shoulder

            run_analyze_shoulder(
                subject=args.subject,
                movement=args.movement,
                manifest_path=args.manifest_path,
                input_root=args.input_root,
                reference_view=args.reference_view,
                **_fused_analysis_kwargs(args),
            )
        elif args.command == "shoulder-pipeline":
            from shoulderlab.shoulder_pipeline import run_shoulder_pipeline

            run_shoulder_pipeline(
                subject=args.subject,
                movement=args.movement,
                reference_view=args.reference_view,
                manifest_path=args.manifest_path,
                input_root=args.input_root,
                **_shoulder_pipeline_kwargs(args),
            )
        elif args.command == "shoulder-dataset":
            from shoulderlab.shoulder_pipeline import run_shoulder_dataset

            run_shoulder_dataset(
                reference_view=args.reference_view,
                manifest_path=args.manifest_path,
                input_root=args.input_root,
                subject=args.subject,
                movement=args.movement,
                continue_on_error=args.continue_on_error,
                **_shoulder_pipeline_kwargs(args),
            )
    except SystemExit as exc:
        if exc.code in (0, None):
            raise
        if isinstance(exc.code, int):
            logger.error("Command exited: %s (code=%s)", args.command, exc.code)
            raise
        logger.error("%s", exc.code)
        raise SystemExit(1) from None
    except Exception:
        logger.exception("Command failed: %s", args.command)
        raise
    else:
        logger.info("Finished command: %s", args.command)


if __name__ == "__main__":
    main()
