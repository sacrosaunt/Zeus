#!/usr/bin/env python3
"""Utility to materialize a Hugging Face model repository locally."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face model repo to a local directory."
    )
    parser.add_argument(
        "--repo-id",
        default="Lightricks/LTX-Video",
        help="Model repository to download (default: %(default)s).",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional git revision (tag/branch/commit) to pin.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token if the model requires authentication.",
    )
    parser.add_argument(
        "--local-dir",
        default="models/LTX-Video",
        help="Destination directory for the model snapshot.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from huggingface_hub import snapshot_download

    destination = Path(args.local_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=str(destination),
        local_dir_use_symlinks=False,
        token=args.token,
    )

    print(f"Model synchronized to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
