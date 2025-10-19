#!/usr/bin/env python3
"""Deployment helper for the Zeus project."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
MODEL_DIR = REPO_ROOT / "models" / "ltxv-2b-0.9.6-distilled"
STATUS_DIR = MODEL_DIR.parent
READY_FLAG = STATUS_DIR / ".server_ready"
BUILDING_FLAG = STATUS_DIR / ".server_building"
LEGACY_READY_FLAG = STATUS_DIR / ".model_ready"
LEGACY_DOWNLOADING_FLAG = STATUS_DIR / ".model_downloading"


def run_command(cmd: list[str], *, cwd: Path | None = None) -> None:
    """Execute a subprocess, streaming output and raising on failure."""
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd or REPO_ROOT)


def model_present() -> bool:
    return MODEL_DIR.exists() and any(MODEL_DIR.iterdir())


def _unlink(path: Path) -> None:
    if path.exists():
        path.unlink()


def mark_building() -> None:
    """Reset flags and mark as 'building'."""
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    _unlink(READY_FLAG)
    _unlink(LEGACY_READY_FLAG)
    BUILDING_FLAG.touch()
    _unlink(LEGACY_DOWNLOADING_FLAG)


def mark_ready() -> None:
    """Reset flags and mark as 'ready'."""
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    _unlink(BUILDING_FLAG)
    _unlink(LEGACY_DOWNLOADING_FLAG)
    READY_FLAG.touch()
    LEGACY_READY_FLAG.touch()


def start_core_services() -> None:
    """Start web tier services to accept traffic while the model syncs."""
    run_command(["docker", "compose", "up", "--build", "-d", "redis", "app1", "app2", "app3", "caddy"])


def start_inference_service() -> None:
    """Start inference once models are ready."""
    run_command(["docker", "compose", "up", "--build", "-d", "inference"])


def download_model() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Model snapshot missing; downloading to {MODEL_DIR}")
    run_command([sys.executable, "scripts/download_model.py", "--local-dir", str(MODEL_DIR)])


def show_status() -> None:
    run_command(["docker", "compose", "ps"])


def main() -> int:
    mark_building()

    already_present = model_present()
    if already_present:
        print(f"Model already present at {MODEL_DIR}")
    else:
        print("Model assets not found; preparing to download.")

    try:
        start_core_services()

        if not already_present:
            download_model()

        start_inference_service()
    except subprocess.CalledProcessError as exc:
        print("Server preparation failed; services will remain in standby.")
        print(exc)
        return 1

    mark_ready()
    show_status()
    print("Deployment complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
