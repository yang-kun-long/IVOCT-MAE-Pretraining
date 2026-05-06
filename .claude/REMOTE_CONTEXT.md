# Remote Server Context

Last verified: 2026-05-06

This file is the quick-start context for future sessions that need to inspect or operate on the remote training server. Do not store passwords, API tokens, or private keys here.

## Local Remote Helper

Use the local helper from the repository root:

```powershell
python scripts/remote_ops.py show-session
python scripts/remote_ops.py exec "cd /root/CN_seg && git status --short --branch"
python scripts/remote_ops.py upload local.txt /root/CN_seg/local.txt
python scripts/remote_ops.py download /root/CN_seg/path/to/file result/path/to/file
```

Session credentials are read in this order:

1. CLI flags: `--host`, `--port`, `--username`, `--password`
2. Environment variables: `REMOTE_HOST`, `REMOTE_PORT`, `REMOTE_USERNAME`, `REMOTE_PASSWORD`
3. Local ignored file: `.claude/remote_session.json`

Preferred setup command, with password entered interactively:

```powershell
python scripts/remote_ops.py save-session --host connect.bjb2.seetacloud.com --port 48198 --username root
```

`.claude/remote_session.json` is intentionally ignored by Git. Keep it local only.

## Server Location And Git State

Remote repository path:

```text
/root/CN_seg
```

Verified server state after sync:

```text
branch: master
HEAD: 017b94f chore: ignore generated server artifacts
remote: origin -> https://gitcode.com/yangkunlong/IVOCT-MAE-Pretraining.git
```

Local repository remote layout after cleanup:

```text
origin  -> https://github.com/yang-kun-long/IVOCT-MAE-Pretraining.git
gitcode -> https://gitcode.com/yangkunlong/IVOCT-MAE-Pretraining.git
```

The temporary sync branch `sync/server-progress-20260506` was merged into `master` and deleted locally and remotely. Maintain `master` unless a future task explicitly needs a feature branch.

## Server Sync History

On 2026-05-06, server-only experiment progress was imported into `master`:

- `65d13c1 sync: import server experiment progress`
- `017b94f chore: ignore generated server artifacts`

During server alignment, pre-pull backups were kept:

```text
/root/CN_seg_pre_pull_backup_20260506_113324
stash@{0}: pre-master-sync tracked changes 20260506_113324
```

The stash is a safety copy of tracked edits that were already incorporated into `master`. Keep it until the next successful training run confirms the synced code works on the server.

## Generated Artifacts

Large or generated files should stay on the server or move to artifact storage, not normal Git:

- `DATA/`
- `DATA_backup_*/`
- `seven/checkpoints_v2/*.pth`
- `seven/seg/checkpoints/*.pth`
- `seven/logs_v2/`
- `seven/recon_vis_v2/`
- `seven/seg/logs/`
- `seven/seg/vis/`
- `seven/seg/*.log`
- `seven/seg/*.pid`
- generated review/export directories under `result/`

These are covered by `.gitignore` after `017b94f`.

## Server Environment Notes

Verified on 2026-05-06:

```text
python: not found in PATH
python3: not found in PATH
usable Python: /root/miniconda3/bin/python
CUDA_VISIBLE_DEVICES: unset/empty
```

Use explicit Python on the server when needed:

```bash
/root/miniconda3/bin/python -m pytest
/root/miniconda3/bin/python seven/seg/train_clean_weighted_4fold_v2.py
```

Remote helper environment variables on the local machine:

- `REMOTE_HOST`
- `REMOTE_PORT`
- `REMOTE_USERNAME`
- `REMOTE_PASSWORD`

## Safe Server Update Pattern

For normal updates after this sync:

```powershell
python scripts/remote_ops.py exec "cd /root/CN_seg && git pull --ff-only origin master && git status --short --branch"
```

Before any risky server cleanup or branch switch, inspect first:

```powershell
python scripts/remote_ops.py exec "cd /root/CN_seg && git status --short --ignored | head -120 && git stash list -n 5"
```

Do not run destructive cleanup against `/root/CN_seg` unless the target paths are explicitly reviewed. Checkpoints and result directories are intentionally preserved.
