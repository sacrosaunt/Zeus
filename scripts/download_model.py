#!/usr/bin/env python3
"""Utility to materialize a Hugging Face model repository locally."""

from __future__ import annotations
from pathlib import Path


REPO_ID = "Lightricks/ltxv-2b-0.9.6-distilled"
LOCAL_DIR = "models/ltxv-2b-0.9.6-distilled"


def main() -> int:
    from huggingface_hub import snapshot_download

    destination = Path(LOCAL_DIR).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    # Always include the configuration files and checkpoints needed to load the model end-to-end.
    allow_patterns: list[str] = [
        "model_index.json",
        "scheduler/*",
        "tokenizer/*",
        "text_encoder/*",
        "transformer/*",
        "vae/*",
        "*.txt",
        "*.py",
        "*.safetensors",
    ]

    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(destination),
        allow_patterns=allow_patterns,
    )

    print(f"Model synchronized to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
