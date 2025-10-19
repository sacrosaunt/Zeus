#!/usr/bin/env python3
"""Utility to materialize a Hugging Face model repository locally."""

from __future__ import annotations
from pathlib import Path


REPO_ID = "Lightricks/LTX-Video"
LOCAL_DIR = "models/ltxv-2b-0.9.6-distilled"


def main() -> int:
    from huggingface_hub import hf_hub_download, snapshot_download

    destination = Path(LOCAL_DIR).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    # Pull only the pieces needed by the official LTX-Video runner.
    allow_patterns: list[str] = [
        "model_index.json",
        "configs/ltxv-2b-0.9.6-distilled.yaml",
        "*.txt",
        "*.py",
        "*.md",
        "ltxv-2b-0.9.6-distilled*.safetensors",
    ]

    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(destination),
        allow_patterns=allow_patterns,
    )

    hf_hub_download(
        repo_id=REPO_ID,
        filename="configs/ltxv-2b-0.9.6-distilled.yaml",
        repo_type="model",
        local_dir=str(destination),
        local_dir_use_symlinks=False,
    )
    hf_hub_download(
        repo_id=REPO_ID,
        filename="ltxv-2b-0.9.6-distilled-04-25.safetensors",
        repo_type="model",
        local_dir=str(destination),
        local_dir_use_symlinks=False,
    )

    print(f"Model synchronized to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
