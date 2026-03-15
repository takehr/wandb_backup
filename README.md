# W&B Export Helpers

Small helper scripts for exporting W&B runs into offline-run directories and syncing them later with `wandb sync`.

## Files

- `export_wandb_run_for_sync.py`: export one or more runs
- `export_wandb_project_for_sync.py`: export every run in a project
- `sync_all_wandb_exports.py`: sync every exported offline-run directory under a root
- `wandb_sync_compat.py`: run `wandb sync` with the protobuf Python runtime forced on

## Requirements

- Python with `wandb` installed
- `WANDB_API_KEY` set, or pass `--api-key`

## Export A Single Run

```bash
python export_wandb_run_for_sync.py entity/project/run_id
```

Examples:

```bash
python export_wandb_run_for_sync.py entity/project/run_id
python export_wandb_run_for_sync.py run_id --entity entity --project project
```

Useful options:

- `--output-dir wandb-export`
- `--overwrite`
- `--no-files`
- `--no-logs`
- `--history-mode sampled`
- `--history-mode validated-sampled`
- `--history-mode full`
- `--history-samples 1000000`
- `--file-download-workers 8`

## Export A Whole Project

```bash
python export_wandb_project_for_sync.py entity/project
```

Useful options:

- `--continue-on-error`
- `--overwrite`
- `--history-mode validated-sampled`
- `--history-samples 1000000`
- `--file-download-workers 8`

## Sync Exported Runs

```bash
python sync_all_wandb_exports.py --exports-dir wandb-export
```

Sync into a different entity / project:

```bash
python /Users/haratakehiro/backup/sync_all_wandb_exports.py \
  /Users/haratakehiro/backup/wandb-export/reppo \
  --entity entity \
  --project new_project
```

Dry-run first:

```bash
python sync_all_wandb_exports.py \
  --exports-dir wandb-export \
  --entity entity \
  --project new_project \
  --dry-run
```

If `wandb sync` has protobuf runtime issues, use:

```bash
python wandb_sync_compat.py sync path/to/offline-run-dir
```

## Recommended History Modes

- `sampled`: lightweight, close to what the W&B UI shows
- `validated-sampled`: sampled history plus sanity checks; good default when `full` is unreliable
- `full`: attempts to export every history row

For runs where exact full-history export is unreliable, a practical command is:

```bash
python export_wandb_run_for_sync.py entity/project/run_id \
  --history-mode validated-sampled \
  --history-samples 1000000 \
  --file-download-workers 8 \
  --overwrite
```

## Known Limitations

- Some runs appear to trigger duplicated rows from `wandb.Api().run(...).scan_history()`. In those cases, `full` can become very large or fail.
- `validated-sampled` is safer than plain `sampled`, but it is still based on W&B sampled history rather than a guaranteed raw-history export.
- This repo includes a NumPy 2.0 compatibility shim for W&B importer code that still references `np.NaN`.
