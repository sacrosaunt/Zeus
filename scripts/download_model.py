#!/usr/bin/env python3
"""Utility to materialize a Hugging Face model repository locally."""

from __future__ import annotations
from pathlib import Path


REPO_ID = "Lightricks/LTX-Video"
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
        "transformer/config.json",
        "vae/*",
        "*.txt",
        "*.py",
        "ltxv-2b-0.9.6-distilled*.safetensors",
    ]

    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(destination),
        allow_patterns=allow_patterns,
    )

    weight_name = "ltxv-2b-0.9.6-distilled-04-25.safetensors"
    weight_file = destination / weight_name
    transformer_dir = destination / "transformer"
    transformer_dir.mkdir(parents=True, exist_ok=True)
    target_weight = transformer_dir / "diffusion_pytorch_model.safetensors"

    if weight_file.exists():
        # ensure the 2B weights sit where diffusers expects them.
        if target_weight.exists():
            target_weight.unlink()
        weight_file.replace(target_weight)
    else:
        print(f"Warning: expected weight file {weight_name} missing in snapshot.")

    print(f"Model synchronized to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
