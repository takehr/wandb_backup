#!/usr/bin/env python3
"""Sync all exported W&B offline-run directories with rewritten run ids."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sync every offline-run export under a directory, overriding the run id "
            "so deleted original ids do not conflict."
        )
    )
    parser.add_argument(
        "exports_dir_arg",
        nargs="?",
        help="Directory containing offline-run-* exports.",
    )
    parser.add_argument(
        "--exports-dir",
        default="/Users/haratakehiro/backup/wandb-export",
        help="Directory containing offline-run-* exports.",
    )
    parser.add_argument(
        "--id-suffix",
        default="-restored",
        help="Suffix appended to each original run id.",
    )
    parser.add_argument(
        "--entity",
        help="Override entity during sync.",
    )
    parser.add_argument(
        "--project",
        help="Override project during sync.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep syncing other runs if one fails.",
    )
    return parser.parse_args()


def manifest_created_at(export_dir: Path) -> dt.datetime | None:
    manifest_path = export_dir / "export-manifest.json"
    if not manifest_path.exists():
        return None

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    created_at = data.get("created_at")
    if not created_at:
        return None

    try:
        return dt.datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError:
        return None


def iter_exports(root: Path):
    exports = [
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith("offline-run-")
    ]
    dated_exports = []
    for path in exports:
        created_at = manifest_created_at(path)
        dated_exports.append(
            (
                created_at or dt.datetime.max.replace(tzinfo=dt.timezone.utc),
                created_at is None,
                path,
            )
        )
    dated_exports.sort(key=lambda item: (item[1], item[0], item[2].name))
    yield from (path for _, _, path in dated_exports)


def find_original_run_id(export_dir: Path) -> str:
    manifest_path = export_dir / "export-manifest.json"
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_id = data.get("run_id")
        if run_id:
            return run_id

    for candidate in export_dir.glob("run-*.wandb"):
        name = candidate.name
        if name.startswith("run-") and name.endswith(".wandb"):
            return name[len("run-") : -len(".wandb")]

    raise RuntimeError(f"Could not determine run id for {export_dir}")


def build_sync_command(
    export_dir: Path,
    run_id: str,
    suffix: str,
    entity: str | None,
    project: str | None,
) -> list[str]:
    cmd = ["wandb", "sync", "--id", f"{run_id}{suffix}"]
    if entity:
        cmd.extend(["--entity", entity])
    if project:
        cmd.extend(["--project", project])
    cmd.append(str(export_dir))
    return cmd


def main() -> int:
    args = parse_args()
    exports_dir = Path(args.exports_dir_arg or args.exports_dir).resolve()
    if not exports_dir.exists():
        print(f"error: exports dir does not exist: {exports_dir}", file=sys.stderr)
        return 1

    env = os.environ.copy()
    env.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

    failed = False
    for export_dir in iter_exports(exports_dir):
        run_id = find_original_run_id(export_dir)
        cmd = build_sync_command(
            export_dir=export_dir,
            run_id=run_id,
            suffix=args.id_suffix,
            entity=args.entity,
            project=args.project,
        )
        print(" ".join(cmd))
        if args.dry_run:
            continue

        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            failed = True
            if not args.continue_on_error:
                return result.returncode

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
