"""Command line interface for ShoulderLab."""

from __future__ import annotations

import argparse
from pathlib import Path

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shoulderlab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Analyze one HSMR .npy/.npz output.")
    analyze.add_argument("-i", "--input-path", type=Path, required=True)
    analyze.add_argument("-o", "--output-dir", type=Path, default=DATA_OUTPUTS / "shoulder_analysis")
    analyze.add_argument("-s", "--side", choices=["right", "left", "both"], default="both")
    _add_analysis_options(analyze)

    analyze_uucm = subparsers.add_parser("analyze-uucm", help="Analyze all UUCM HSMR .npy outputs.")
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

    q2_summary = subparsers.add_parser("q2-summary", help="Summarize Q2 temporal feature JSON outputs.")
    q2_summary.add_argument("input_dir", type=Path, nargs="?", default=DATA_OUTPUTS / "UUCM" / "q2_analysis")
    q2_summary.add_argument("--docs-path", type=Path, default=SHOULDERLAB_ROOT / "docs" / "Q2_Temporal_Feature_Noise_Report.md")

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

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
    elif args.command == "q2-summary":
        from shoulderlab.q2_summary import summarize_q2

        summarize_q2(input_dir=args.input_dir, docs_path=args.docs_path)


if __name__ == "__main__":
    main()
