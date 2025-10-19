#!/usr/bin/env python3
"""Utility to materialize a Hugging Face model repository locally."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_REPO_ID = "Lightricks/LTX-Video"
DEFAULT_LOCAL_DIR = "models/ltxv-2b-0.9.6"
DEFAULT_CHECKPOINT = "ltxv-2b-0.9.6-dev-04-25.safetensors"

ALLOW_PATTERNS: list[str] = [
    "model_index.json",
    "configs/ltxv-2b-0.9.6-dev.yaml",
    "*.txt",
    "*.py",
    "*.md",
    "ltxv-2b-0.9.6-dev*.safetensors",
]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Hugging Face repository identifier (default: %(default)s)",
    )
    parser.add_argument(
        "--local-dir",
        default=DEFAULT_LOCAL_DIR,
        help="Destination directory for model assets (default: %(default)s)",
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Specific checkpoint filename to ensure is downloaded (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from huggingface_hub import hf_hub_download, snapshot_download

    args = _parse_args(argv)

    destination = Path(args.local_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(destination),
        allow_patterns=ALLOW_PATTERNS,
    )

    checkpoint_path = destination / args.checkpoint
    if not checkpoint_path.exists():
        hf_hub_download(
            repo_id=args.repo_id,
            filename=args.checkpoint,
            repo_type="model",
            local_dir=str(destination),
            local_dir_use_symlinks=False,
        )

    print(f"Model synchronized to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
