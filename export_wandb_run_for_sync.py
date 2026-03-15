#!/usr/bin/env python3
"""Export W&B cloud runs into local offline-run directories that `wandb sync` can replay."""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import wandb
import numpy as np
from wandb.apis.importers.internals import internal
from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.interface.interface_queue import InterfaceQueue
from wandb.sdk.internal.datastore import DataStore
import yaml

if not hasattr(np, "NaN"):
    np.NaN = np.nan


class SyncExportError(RuntimeError):
    pass


class ExportableWandbRun:
    def __init__(
        self,
        run,
        *,
        api: wandb.Api,
        export_run_dir: Path,
        history_mode: str,
        history_samples: int,
        file_download_workers: int,
    ) -> None:
        self.run = run
        self.api = api
        self._export_run_dir = export_run_dir
        self._history_mode = history_mode
        self._history_samples = history_samples
        self._file_download_workers = file_download_workers
        self._files: list[tuple[str, str]] | None = None
        self._metadata: dict | None = None
        self._config_yaml: dict | None = None
        self._seen_full_history_rows: set[str] = set()
        self._cached_sampled_history_rows: list[dict] | None = None

    def run_id(self) -> str:
        return self.run.id

    def entity(self) -> str:
        return self.run.entity

    def project(self) -> str:
        return self.run.project

    def config(self) -> dict:
        return dict(self.run.config or {})

    def summary(self) -> dict:
        return dict(self.run.summary or {})

    def metrics(self):
        start_time_seconds = self.start_time() / 1000.0
        previous: dict[str, object] = {}
        always_emit = {"_step", "_timestamp", "_runtime"}
        for row in self._history_rows():
            self._validate_full_history_row(row)
            clean = {k: v for k, v in row.items() if v is not None}
            runtime = clean.get("_runtime")
            timestamp = clean.get("_timestamp")

            if runtime is not None and timestamp is None:
                clean["_timestamp"] = start_time_seconds + float(runtime)
            elif timestamp is not None and runtime is None:
                clean["_runtime"] = float(timestamp) - start_time_seconds

            delta = {
                key: value
                for key, value in clean.items()
                if key in always_emit or previous.get(key) != value
            }
            previous = clean
            yield delta

    def _validate_full_history_row(self, row: dict) -> None:
        if self._history_mode != "full":
            return
        signature = json.dumps(row, sort_keys=True, default=str)
        if signature in self._seen_full_history_rows:
            step = row.get("_step")
            raise SyncExportError(
                "W&B scan_history() returned a duplicate history row while exporting "
                f"full history (step={step!r}). Aborting because continuing would "
                "silently produce a non-faithful export. Use sampled history or "
                "investigate the upstream run/API behavior."
            )
        self._seen_full_history_rows.add(signature)

    def _history_rows(self):
        if self._history_mode in {"sampled", "validated-sampled"}:
            yield from self._sampled_history_rows()
            return
        yield from self.run.scan_history()

    def _sampled_history_rows(self) -> list[dict]:
        if self._cached_sampled_history_rows is None:
            rows = list(self.run.history(samples=self._history_samples, pandas=False))
            if self._history_mode == "validated-sampled":
                self._validate_sampled_history_rows(rows)
            self._cached_sampled_history_rows = rows
        return self._cached_sampled_history_rows

    def _validate_sampled_history_rows(self, rows: list[dict]) -> None:
        if not rows:
            return
        if len(rows) >= self._history_samples:
            raise SyncExportError(
                "Validated sampled history hit the sample limit exactly. Aborting because "
                "the result may still be truncated; increase --history-samples."
            )

        seen_steps: set[object] = set()
        prev_step: float | None = None
        numeric_steps = 0
        for row in rows:
            step = row.get("_step")
            if step is None:
                continue
            if step in seen_steps:
                raise SyncExportError(
                    f"Validated sampled history contains a duplicate _step ({step!r})."
                )
            seen_steps.add(step)
            if isinstance(step, (int, float)) and not isinstance(step, bool):
                step_value = float(step)
                if prev_step is not None and step_value <= prev_step:
                    raise SyncExportError(
                        "Validated sampled history is not strictly increasing in _step."
                    )
                prev_step = step_value
                numeric_steps += 1

        expected_last_step = self.summary().get("_step")
        if (
            expected_last_step is not None
            and numeric_steps > 0
            and prev_step is not None
            and float(expected_last_step) != prev_step
        ):
            raise SyncExportError(
                "Validated sampled history does not reach the run summary _step. "
                f"Expected {expected_last_step!r}, got {prev_step!r}."
            )

    def run_group(self) -> str | None:
        return self.run.group

    def job_type(self) -> str | None:
        return self.run.job_type

    def display_name(self) -> str:
        return self.run.display_name or self.run.name or self.run.id

    def notes(self) -> str:
        return self.run.notes or ""

    def tags(self) -> list[str]:
        return list(self.run.tags or [])

    def artifacts(self):
        return None

    def used_artifacts(self):
        return None

    def os_version(self) -> str | None:
        return self._metadata_file().get("os")

    def python_version(self) -> str | None:
        return self._metadata_file().get("python")

    def cuda_version(self) -> str | None:
        return self._metadata_file().get("cuda")

    def program(self) -> str | None:
        return self._metadata_file().get("program")

    def host(self) -> str | None:
        return self._metadata_file().get("host")

    def username(self) -> str | None:
        user = getattr(self.run, "user", None)
        return getattr(user, "username", None)

    def executable(self) -> str | None:
        return self._metadata_file().get("executable")

    def gpus_used(self):
        return None

    def cpus_used(self):
        return None

    def memory_used(self):
        return None

    def runtime(self) -> int | None:
        wandb_runtime = self.run.summary.get("_wandb", {}).get("runtime")
        base_runtime = self.run.summary.get("_runtime")
        value = wandb_runtime if wandb_runtime is not None else base_runtime
        return int(value) if value is not None else None

    def start_time(self) -> int:
        created_at = dt.datetime.fromisoformat(self.run.created_at.replace("Z", "+00:00"))
        return int(created_at.timestamp() * 1000)

    def code_path(self) -> str | None:
        path = self._metadata_file().get("codePath")
        return f"code/{path}" if path else None

    def cli_version(self) -> str | None:
        config = self._config_file()
        return config.get("_wandb", {}).get("value", {}).get("cli_version")

    def files(self):
        if self._files is None:
            files_dir = self._export_run_dir / "files"
            files_dir.mkdir(parents=True, exist_ok=True)
            try:
                file_objs = [
                    file_obj
                    for file_obj in self.run.files()
                    if file_obj.size != 0 and "wandb_manifest.json.deadlist" not in file_obj.name
                ]
            except TypeError:
                file_objs = []
            downloaded = self._download_files(file_objs, files_dir)
            self._files = downloaded

        yield from self._files

    def _download_files(self, file_objs, files_dir: Path) -> list[tuple[str, str]]:
        if not file_objs:
            return []
        if self._file_download_workers <= 1:
            return [self._download_one_file(file_obj, files_dir) for file_obj in file_objs]

        downloaded_by_name: dict[str, tuple[str, str]] = {}
        with ThreadPoolExecutor(max_workers=self._file_download_workers) as executor:
            future_to_name = {
                executor.submit(self._download_one_file, file_obj, files_dir): file_obj.name
                for file_obj in file_objs
            }
            for future in as_completed(future_to_name):
                file_name = future_to_name[future]
                downloaded_by_name[file_name] = future.result()

        return [downloaded_by_name[file_obj.name] for file_obj in file_objs]

    def _download_one_file(self, file_obj, files_dir: Path) -> tuple[str, str]:
        result = file_obj.download(str(files_dir), exist_ok=True, api=self.api)
        return (result.name, "end")

    def logs(self):
        for path, _ in self.files():
            if path.endswith("output.log"):
                with open(path, encoding="utf-8", errors="replace") as handle:
                    yield from handle.readlines()

    def _metadata_file(self) -> dict:
        if self._metadata is None:
            self._metadata = self._load_json_file("wandb-metadata.json")
        return self._metadata

    def _config_file(self) -> dict:
        if self._config_yaml is None:
            config_path = self._find_downloaded_file("config.yaml")
            if config_path is None:
                self._config_yaml = {}
            else:
                self._config_yaml = yaml.safe_load(Path(config_path).read_text()) or {}
        return self._config_yaml

    def _find_downloaded_file(self, suffix: str) -> str | None:
        for path, _ in self.files():
            if path.endswith(suffix):
                return path
        return None

    def _load_json_file(self, suffix: str) -> dict:
        file_path = self._find_downloaded_file(suffix)
        if file_path is None:
            return {}
        with open(file_path, encoding="utf-8") as handle:
            return json.load(handle)


class ExportRecordMaker(internal.RecordMaker):
    def __init__(self, run: ExportableWandbRun, export_run_dir: Path) -> None:
        super().__init__(run=run)
        self._export_run_dir = export_run_dir

    @property
    def run_dir(self) -> str:
        self._export_run_dir.mkdir(parents=True, exist_ok=True)
        return str(self._export_run_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export cloud W&B runs into local offline-run directories that can be "
            "uploaded later with `wandb sync`."
        )
    )
    parser.add_argument(
        "runs",
        nargs="+",
        help="Run paths in the form entity/project/run_id.",
    )
    parser.add_argument(
        "--output-dir",
        default="wandb-export",
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
        "--entity",
        help="Default entity for run ids passed as project/run_id.",
    )
    parser.add_argument(
        "--project",
        help="Default project for run ids passed as run_id.",
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
    return parser.parse_args()


def normalize_run_path(raw: str, default_entity: str | None, default_project: str | None) -> str:
    parts = [part for part in raw.strip("/").split("/") if part]
    if len(parts) == 3:
        return "/".join(parts)
    if len(parts) == 2:
        if not default_entity:
            raise SyncExportError(
                f"Run path '{raw}' is missing the entity component; pass --entity."
            )
        return "/".join([default_entity, parts[0], parts[1]])
    if len(parts) == 1:
        if not default_entity or not default_project:
            raise SyncExportError(
                f"Run path '{raw}' must be entity/project/run_id or require both --entity and --project."
            )
        return "/".join([default_entity, default_project, parts[0]])
    raise SyncExportError(f"Invalid run path '{raw}'.")


def build_api(api_key: str | None, base_url: str) -> wandb.Api:
    if not api_key:
        raise SyncExportError("WANDB_API_KEY is required.")
    return wandb.Api(api_key=api_key, overrides={"base_url": base_url})


def created_at_to_stamp(created_at: str) -> str:
    when = dt.datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    return when.strftime("%Y%m%d_%H%M%S")


def export_dir_name(run) -> str:
    return f"offline-run-{created_at_to_stamp(run.created_at)}-{run.id}"


def make_record_writer(target: Path) -> DataStore:
    target.parent.mkdir(parents=True, exist_ok=True)
    datastore = DataStore()
    datastore.open_for_write(str(target))
    return datastore


def serialize_records(datastore: DataStore, records: Iterable[pb.Record]) -> None:
    for record in records:
        datastore._write_data(record.SerializeToString())

    datastore._fp.flush()
    os.fsync(datastore._fp.fileno())
    datastore._fp.close()


def make_exit_and_final_records(
    run_wrapper: ExportableWandbRun, interface: InterfaceQueue
) -> list[pb.Record]:
    exit_record = pb.RunExitRecord()
    exit_record.exit_code = 0
    runtime = run_wrapper.runtime()
    if runtime is not None:
        exit_record.runtime = runtime

    final_record = pb.FinalRecord()
    return [
        interface._make_record(exit=exit_record),
        interface._make_record(final=final_record),
    ]


def write_manifest(
    export_root: Path,
    run_path: str,
    run,
    *,
    history_mode: str,
    history_samples: int,
) -> None:
    manifest = {
        "source_run": run_path,
        "exported_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "entity": run.entity,
        "project": run.project,
        "run_id": run.id,
        "run_name": run.name,
        "run_url": run.url,
        "created_at": run.created_at,
        "history_mode": history_mode,
    }
    if history_mode in {"sampled", "validated-sampled"}:
        manifest["history_samples"] = history_samples
    (export_root / "export-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def export_run(
    api: wandb.Api,
    run_path: str,
    output_dir: Path,
    include_files: bool,
    include_logs: bool,
    history_mode: str,
    history_samples: int,
    file_download_workers: int,
    overwrite: bool,
) -> Path:
    public_run = api.run(run_path)
    run_dir = output_dir / export_dir_name(public_run)

    if run_dir.exists():
        if not overwrite:
            raise SyncExportError(
                f"Export directory already exists: {run_dir}. Pass --overwrite to replace it."
            )
        shutil.rmtree(run_dir)

    run_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(
        run_dir,
        run_path,
        public_run,
        history_mode=history_mode,
        history_samples=history_samples,
    )

    wrapped_run = ExportableWandbRun(
        public_run,
        api=api,
        export_run_dir=run_dir,
        history_mode=history_mode,
        history_samples=history_samples,
        file_download_workers=file_download_workers,
    )
    record_maker = ExportRecordMaker(wrapped_run, export_run_dir=run_dir)
    run_file = run_dir / f"run-{public_run.id}.wandb"
    datastore = make_record_writer(run_file)
    records = record_maker.make_records(
        internal.SendManagerConfig(
            files=include_files,
            media=include_files,
            code=include_files,
            history=True,
            summary=True,
            terminal_output=include_logs,
        )
    )
    serialize_records(
        datastore,
        itertools.chain(
            records,
            make_exit_and_final_records(wrapped_run, record_maker.interface),
        ),
    )

    return run_dir


def main() -> int:
    args = parse_args()
    try:
        if args.history_samples <= 0:
            raise SyncExportError("--history-samples must be a positive integer.")
        if args.file_download_workers <= 0:
            raise SyncExportError("--file-download-workers must be a positive integer.")
        api = build_api(args.api_key, args.base_url)
        output_dir = Path(args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        exported: list[Path] = []
        for raw_run in args.runs:
            run_path = normalize_run_path(raw_run, args.entity, args.project)
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
            exported.append(target_dir)
            print(target_dir)
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
