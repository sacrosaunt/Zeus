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
    parser.add_argument(
        "--weights-file",
        action="append",
        default=["ltx-video-2b-v0.9.safetensors"],
        help=(
            "Specific weight files to download. "
            "Provide multiple times to fetch more than one file."
        ),
    )
    parser.add_argument(
        "--include-config",
        dest="include_config",
        action="store_true",
        default=True,
        help="Fetch lightweight config/tokenizer files required for full inference (default).",
    )
    parser.add_argument(
        "--no-include-config",
        dest="include_config",
        action="store_false",
        help="Skip downloading config/tokenizer assets.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from huggingface_hub import hf_hub_download, snapshot_download

    destination = Path(args.local_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    if args.include_config:
        snapshot_download(
            repo_id=args.repo_id,
            revision=args.revision,
            local_dir=str(destination),
            token=args.token,
            allow_patterns=[
                "model_index.json",
                "scheduler/*",
                "tokenizer/*",
                "text_encoder/*",
                "transformer/*",
                "vae/*",
                "*.txt",
                "*.py",
            ],
        )

    weights = [w for w in (args.weights_file or []) if w]
    if not weights:
        print("No weights specified; nothing to download.")
    for weight in weights:
        print(f"Downloading weight file {weight}")
        hf_hub_download(
            repo_id=args.repo_id,
            revision=args.revision,
            filename=weight,
            local_dir=str(destination),
            token=args.token,
        )

    print(f"Model synchronized to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
