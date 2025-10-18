#!/usr/bin/env python3
"""Deployment helper for the Zeus project."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
MODEL_DIR = REPO_ROOT / "models" / "LTX-Video"


def run_command(cmd: list[str], *, cwd: Path | None = None) -> None:
    """Execute a subprocess, streaming output and raising on failure."""
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd or REPO_ROOT)


def ensure_model_present() -> None:
    """Download the model snapshot if it is missing or empty."""
    if MODEL_DIR.exists() and any(MODEL_DIR.iterdir()):
        print(f"Model already present at {MODEL_DIR}")
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Model snapshot missing; downloading to {MODEL_DIR}")
    run_command([sys.executable, "scripts/download_model.py", "--local-dir", str(MODEL_DIR)])


def start_services() -> None:
    """Start Redis first, then the rest of the stack."""
    # Ensure Redis is up before bringing the rest of the services online.
    run_command(["docker", "compose", "up", "-d", "redis"])
    run_command(["docker", "compose", "up", "--build", "-d"])
    run_command(["docker", "compose", "ps"])


def main() -> int:
    ensure_model_present()
    start_services()
    print("Deployment complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
