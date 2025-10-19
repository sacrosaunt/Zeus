#!/usr/bin/env python3
"""Utility to materialize a Hugging Face model repository locally."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ID = "Lightricks/LTX-Video-0.9.8-13B-distilled"
LOCAL_DIR = "models/ltx-video-0.9.8-13b-distilled"
WEIGHTS: tuple[str, ...] | None = None


def main() -> int:
    from huggingface_hub import hf_hub_download, snapshot_download

    destination = Path(LOCAL_DIR).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    weights = tuple(filter(None, WEIGHTS or ()))

    allow_patterns: list[str] = [
        "model_index.json",
        "scheduler/*",
        "tokenizer/*",
        "text_encoder/*",
        "transformer/*",
        "vae/*",
        "*.txt",
        "*.py",
    ]
    if not weights:
        allow_patterns.append("*.safetensors")

    snapshot_kwargs = {
        "repo_id": REPO_ID,
        "local_dir": str(destination),
    }
    if allow_patterns:
        snapshot_kwargs["allow_patterns"] = allow_patterns
    snapshot_download(**snapshot_kwargs)

    for weight in weights:
        print(f"Downloading weight file {weight}")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=weight,
            local_dir=str(destination),
        )

    print(f"Model synchronized to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
