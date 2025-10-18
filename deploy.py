#!/usr/bin/env python3
"""Deployment helper for the Zeus project."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
MODEL_DIR = REPO_ROOT / "models" / "LTX-Video"
STATUS_DIR = MODEL_DIR.parent
READY_FLAG = STATUS_DIR / ".model_ready"
DOWNLOADING_FLAG = STATUS_DIR / ".model_downloading"


def run_command(cmd: list[str], *, cwd: Path | None = None) -> None:
    """Execute a subprocess, streaming output and raising on failure."""
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd or REPO_ROOT)


def model_present() -> bool:
    return MODEL_DIR.exists() and any(MODEL_DIR.iterdir())


def mark_downloading() -> None:
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    if READY_FLAG.exists():
        READY_FLAG.unlink()
    DOWNLOADING_FLAG.touch()


def mark_ready() -> None:
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    if DOWNLOADING_FLAG.exists():
        DOWNLOADING_FLAG.unlink()
    READY_FLAG.touch()


def start_core_services() -> None:
    """Start web tier services to accept traffic while the model syncs."""
    run_command(["docker", "compose", "up", "--build", "-d", "redis", "app1", "app2", "caddy"])


def start_inference_service() -> None:
    run_command(["docker", "compose", "up", "--build", "-d", "inference"])


def download_model() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Model snapshot missing; downloading to {MODEL_DIR}")
    run_command([sys.executable, "scripts/download_model.py", "--local-dir", str(MODEL_DIR)])


def show_status() -> None:
    run_command(["docker", "compose", "ps"])


def main() -> int:
    already_present = model_present()
    if already_present:
        print(f"Model already present at {MODEL_DIR}")
        mark_ready()
    else:
        print("Model assets not found; marking as downloading.")
        mark_downloading()

    start_core_services()

    if not already_present:
        try:
            download_model()
            mark_ready()
        except subprocess.CalledProcessError as exc:
            print("Model download failed; services will remain in standby.")
            print(exc)
            return 1

    start_inference_service()
    show_status()
    print("Deployment complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
