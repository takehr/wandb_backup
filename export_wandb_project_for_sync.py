#!/usr/bin/env python3
"""Export all runs in a W&B project into local offline-run directories."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from export_wandb_run_for_sync import build_api, export_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export every run in a W&B project into local offline-run directories "
            "that can later be uploaded with `wandb sync`."
        )
    )
    parser.add_argument(
        "project_path",
        help="Project path in the form entity/project.",
    )
    parser.add_argument(
        "--output-dir",
        default="/Users/haratakehiro/backup/wandb-export",
        help="Directory where offline-run-* exports are created.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai"),
        help="W&B API base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("WANDB_API_KEY"),
        help="W&B API key. Defaults to WANDB_API_KEY.",
    )
    parser.add_argument(
        "--api-timeout",
        type=int,
        default=int(os.environ.get("WANDB_API_TIMEOUT", "60")),
        help="W&B public API timeout in seconds.",
    )
    parser.add_argument(
        "--no-files",
        action="store_true",
        help="Skip downloading run files. The export will be less complete.",
    )
    parser.add_argument(
        "--no-logs",
        action="store_true",
        help="Skip reconstructing console output records.",
    )
    parser.add_argument(
        "--history-mode",
        choices=("sampled", "validated-sampled", "full"),
        default="sampled",
        help=(
            "History export mode. 'sampled' matches the lighter sampled history used by "
            "the W&B UI; 'validated-sampled' fetches sampled history and aborts if the "
            "result looks truncated or inconsistent; 'full' exports every history row."
        ),
    )
    parser.add_argument(
        "--history-samples",
        type=int,
        default=500,
        help="Number of sampled history rows to export when --history-mode=sampled.",
    )
    parser.add_argument(
        "--file-download-workers",
        type=int,
        default=1,
        help="Number of parallel workers to use when downloading run files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing export directory for the same run.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep exporting other runs if one fails.",
    )
    return parser.parse_args()


def normalize_project_path(raw: str) -> str:
    parts = [part for part in raw.strip("/").split("/") if part]
    if len(parts) != 2:
        raise ValueError("project_path must be in the form entity/project")
    return "/".join(parts)


def main() -> int:
    args = parse_args()
    try:
        if args.history_samples <= 0:
            raise ValueError("--history-samples must be a positive integer.")
        if args.file_download_workers <= 0:
            raise ValueError("--file-download-workers must be a positive integer.")
        if args.api_timeout <= 0:
            raise ValueError("--api-timeout must be a positive integer.")
        project_path = normalize_project_path(args.project_path)
        api = build_api(args.api_key, args.base_url, args.api_timeout)
        output_dir = Path(args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        failed = False
        for run in api.runs(project_path):
            run_path = f"{run.entity}/{run.project}/{run.id}"
            print(run_path)
            try:
                target_dir = export_run(
                    api=api,
                    run_path=run_path,
                    output_dir=output_dir,
                    include_files=not args.no_files,
                    include_logs=not args.no_logs,
                    history_mode=args.history_mode,
                    history_samples=args.history_samples,
                    file_download_workers=args.file_download_workers,
                    overwrite=args.overwrite,
                )
                print(target_dir)
            except Exception as exc:
                failed = True
                print(f"error: {run_path}: {exc}", file=sys.stderr)
                if not args.continue_on_error:
                    return 1

        return 1 if failed else 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
