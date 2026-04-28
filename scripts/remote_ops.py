#!/usr/bin/env python
"""Local helper for SSH command execution and SFTP file transfer.

This script is meant to run on the local machine, not on the server.
It stores session credentials only in a local ignored file:

    .claude/remote_session.json

Examples:

    python scripts/remote_ops.py save-session --host connect.example.com --port 22 --username root
    python scripts/remote_ops.py exec "cd /root/CN_seg && git pull"
    python scripts/remote_ops.py download /root/CN_seg/result/file.tar.gz result/file.tar.gz
    python scripts/remote_ops.py upload local.txt /root/remote.txt

NOTE (Git Bash / MSYS2): Git Bash auto-converts Unix paths like /root/... to Windows paths.
Prefix commands with MSYS_NO_PATHCONV=1 to prevent this:

    MSYS_NO_PATHCONV=1 python scripts/remote_ops.py upload local.txt /root/remote.txt
    MSYS_NO_PATHCONV=1 python scripts/remote_ops.py exec "ls /root"
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import posixpath
import sys
from pathlib import Path
from typing import Any

import paramiko


REPO_ROOT = Path(__file__).resolve().parent.parent
SESSION_PATH = REPO_ROOT / ".claude" / "remote_session.json"

# Git Bash on Windows converts /root/... to <git-install>/root/...
# Detect and undo that conversion so users don't need MSYS_NO_PATHCONV=1.
_MSYS_PREFIX_RE = None

def _fix_remote_path(p: str) -> str:
    """Undo Git Bash MSYS path conversion on remote (Linux) paths."""
    import re
    global _MSYS_PREFIX_RE
    if _MSYS_PREFIX_RE is None:
        _MSYS_PREFIX_RE = re.compile(r'^[A-Za-z]:[/\\].*?[/\\]Git[/\\](.*)')
    m = _MSYS_PREFIX_RE.match(p)
    if m:
        return "/" + m.group(1).replace("\\", "/")
    return p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--host", help="SSH hostname")
    common.add_argument("--port", type=int, help="SSH port")
    common.add_argument("--username", help="SSH username")
    common.add_argument("--password", help="SSH password; omit to prompt securely")
    common.add_argument(
        "--session-file",
        default=str(SESSION_PATH),
        help="Path to local session json file",
    )

    save_session = subparsers.add_parser("save-session", parents=[common], help="Save local session config")
    save_session.add_argument(
        "--no-password-prompt",
        action="store_true",
        help="Do not prompt for password if not provided",
    )

    subparsers.add_parser("show-session", parents=[common], help="Show current session without password")

    exec_cmd = subparsers.add_parser("exec", parents=[common], help="Run a remote shell command")
    exec_cmd.add_argument("remote_command", help="Shell command to run on the remote host")

    download = subparsers.add_parser("download", parents=[common], help="Download a file with SFTP")
    download.add_argument("remote_path", help="Remote file path")
    download.add_argument("local_path", help="Local destination path")

    upload = subparsers.add_parser("upload", parents=[common], help="Upload a file with SFTP")
    upload.add_argument("local_path", help="Local file path")
    upload.add_argument("remote_path", help="Remote destination path")

    return parser.parse_args()


def _postprocess_args(args: argparse.Namespace) -> argparse.Namespace:
    """Fix Git Bash MSYS path conversion on any remote_path argument."""
    if hasattr(args, "remote_path"):
        args.remote_path = _fix_remote_path(args.remote_path)
    return args


def session_file_from_args(args: argparse.Namespace) -> Path:
    return Path(args.session_file).expanduser().resolve()


def load_json_session(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def merge_session(args: argparse.Namespace, *, prompt_for_password: bool) -> dict[str, Any]:
    session_path = session_file_from_args(args)
    file_session = load_json_session(session_path)

    merged = {
        "host": args.host or os.getenv("REMOTE_HOST") or file_session.get("host"),
        "port": args.port or _env_int("REMOTE_PORT") or file_session.get("port") or 22,
        "username": args.username or os.getenv("REMOTE_USERNAME") or file_session.get("username"),
        "password": args.password or os.getenv("REMOTE_PASSWORD") or file_session.get("password"),
    }

    missing = [key for key in ("host", "username") if not merged.get(key)]
    if missing:
        raise SystemExit(f"Missing session fields: {', '.join(missing)}")

    if prompt_for_password and not merged.get("password"):
        merged["password"] = getpass.getpass("Remote password: ")

    return merged


def _env_int(name: str) -> int | None:
    raw = os.getenv(name)
    return int(raw) if raw else None


def save_session(args: argparse.Namespace) -> int:
    session_path = session_file_from_args(args)
    session_path.parent.mkdir(parents=True, exist_ok=True)

    merged = merge_session(args, prompt_for_password=not args.no_password_prompt)
    session_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")

    redacted = {**merged, "password": "***" if merged.get("password") else None}
    print(f"Saved session to {session_path}")
    print(json.dumps(redacted, indent=2))
    return 0


def show_session(args: argparse.Namespace) -> int:
    session_path = session_file_from_args(args)
    merged = merge_session(args, prompt_for_password=False)
    redacted = {**merged, "password": "***" if merged.get("password") else None}
    print(f"Session file: {session_path}")
    print(json.dumps(redacted, indent=2))
    return 0


def ssh_client(session: dict[str, Any]) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=session["host"],
        port=int(session["port"]),
        username=session["username"],
        password=session.get("password"),
        look_for_keys=False,
        allow_agent=False,
        timeout=30,
    )
    return client


def exec_remote(args: argparse.Namespace) -> int:
    session = merge_session(args, prompt_for_password=True)
    client = ssh_client(session)
    try:
        stdin, stdout, stderr = client.exec_command(args.remote_command)
        stdin.close()
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        code = stdout.channel.recv_exit_status()
    finally:
        client.close()

    if out:
        print(out, end="" if out.endswith("\n") else "\n")
    if err:
        print(err, end="" if err.endswith("\n") else "\n", file=sys.stderr)
    return code


def ensure_remote_dirs(sftp: paramiko.SFTPClient, remote_path: str) -> None:
    remote_dir = posixpath.dirname(remote_path)
    if not remote_dir:
        return

    current = ""
    for part in remote_dir.strip("/").split("/"):
        current = f"{current}/{part}" if current else f"/{part}"
        try:
            sftp.stat(current)
        except (FileNotFoundError, IOError, OSError):
            try:
                sftp.mkdir(current)
            except (FileNotFoundError, IOError, OSError):
                pass  # already exists or permission on parent dir


def download_file(args: argparse.Namespace) -> int:
    session = merge_session(args, prompt_for_password=True)
    local_path = Path(args.local_path).expanduser().resolve()
    local_path.parent.mkdir(parents=True, exist_ok=True)

    client = ssh_client(session)
    try:
        with client.open_sftp() as sftp:
            sftp.get(args.remote_path, str(local_path))
    finally:
        client.close()

    print(f"Downloaded {args.remote_path} -> {local_path}")
    return 0


def upload_file(args: argparse.Namespace) -> int:
    session = merge_session(args, prompt_for_password=False)
    local_path = Path(args.local_path).expanduser().resolve()
    if not local_path.exists():
        raise SystemExit(f"Local file does not exist: {local_path}")

    client = ssh_client(session)
    try:
        remote_dir = posixpath.dirname(args.remote_path)
        if remote_dir:
            stdin, stdout, stderr = client.exec_command(f"mkdir -p {remote_dir}")
            stdin.close()
            stdout.read()
            stdout.channel.recv_exit_status()
        with client.open_sftp() as sftp:
            sftp.put(str(local_path), args.remote_path)
    finally:
        client.close()

    print(f"Uploaded {local_path} -> {args.remote_path}")
    return 0


def main() -> int:
    args = _postprocess_args(parse_args())
    if args.command == "save-session":
        return save_session(args)
    if args.command == "show-session":
        return show_session(args)
    if args.command == "exec":
        return exec_remote(args)
    if args.command == "download":
        return download_file(args)
    if args.command == "upload":
        return upload_file(args)
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
