#!/usr/bin/env python3
"""Utility to materialize a Hugging Face model repository locally."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Iterable, Set


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


def _configure_hf_environment(destination: Path) -> Path:
    """Ensure Hugging Face cache directories are rooted alongside the model."""

    default_hf_home = destination.parent / "hf-cache"
    hf_home = Path(os.environ.get("HF_HOME") or default_hf_home).expanduser().resolve()
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home / "transformers"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))

    for subdir in ("", "hub", "transformers", "datasets"):
        (hf_home / subdir).mkdir(parents=True, exist_ok=True)
    return hf_home


def _extract_repo_ids(config_paths: Iterable[Path]) -> Set[str]:
    """Scan pipeline config files for remote repositories we should prefetch."""

    candidates: Set[str] = set()

    key_pattern = re.compile(r"model_name_or_path|tokenizer_name_or_path|repo_id", re.IGNORECASE)

    for path in config_paths:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue

        for raw_line in text.splitlines():
            # drop inline comments and whitespace
            line = raw_line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue

            key, value = line.split(":", 1)
            key = key.strip()
            if not key_pattern.search(key):
                continue

            value = value.strip().strip("\"'").strip()
            if not value or value.lower() in {"null", "none"}:
                continue

            # ignore explicit local paths
            if value.startswith(("./", "../", "/", "~")):
                continue
            if Path(value).suffix in {".safetensors", ".pt", ".bin", ".json", ".yaml", ".yml"}:
                continue
            if value.endswith(("/", "\\")):
                value = value.rstrip("/\\")

            candidates.add(value)

    return candidates


def main(argv: list[str] | None = None) -> int:
    from huggingface_hub import hf_hub_download, snapshot_download

    args = _parse_args(argv)

    destination = Path(args.local_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    hf_home = _configure_hf_environment(destination)
    print(f"Using Hugging Face cache at {hf_home}")

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

    config_dir = destination / "configs"
    config_paths = sorted(
        path
        for pattern in ("*.yml", "*.yaml")
        for path in config_dir.glob(pattern)
    ) if config_dir.exists() else []

    repo_ids = _extract_repo_ids(config_paths)
    if repo_ids:
        print("Prefetching dependent repositories to avoid runtime downloads:")
        for repo_id in sorted(repo_ids):
            try:
                print(f"  - {repo_id}")
                snapshot_download(repo_id=repo_id, repo_type="model")
            except Exception as exc:  # noqa: BLE001
                print(f"    ! Failed to prefetch {repo_id}: {exc}")
    else:
        print("No additional repositories detected in pipeline config.")

    print(f"Model synchronized to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
