import json
import logging
import os
import time
from dataclasses import dataclass
from importlib.resources import files as pkg_files
from pathlib import Path
from collections.abc import Callable

import imageio.v2 as imageio
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from ltx_video.inference import (
    calculate_padding,
    create_ltx_video_pipeline,
    get_device as ltx_get_device,
    load_pipeline_config,
    seed_everething,
)
from redis import Redis

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@dataclass
class WorkerConfig:
    redis_url: str
    redis_queue_key: str
    redis_status_key: str
    generated_root: Path
    model_path: Path
    device: str
    frames: int
    fps: int
    height: int
    width: int
    inference_steps: int


def _format_status(state: str, percent: int | None = None) -> str:
    if percent is None:
        return state
    bounded = max(0, min(100, percent))
    return f"{state}:{bounded}"


def load_config() -> WorkerConfig:
    """Load worker configuration from environment variables."""

    def _get(key: str) -> str | None:
        return os.environ.get(key)

    redis_url = _get("REDIS_URL")
    redis_queue_key = _get("REDIS_QUEUE_KEY")
    redis_status_key = _get("REDIS_STATUS_KEY")
    generated_root_value = _get("GENERATED_ROOT")
    model_path_value = _get("LTX_MODEL_ID")
    device = _get("LTX_DEVICE")

    missing = [
        key
        for key, value in {
            "REDIS_URL": redis_url,
            "REDIS_QUEUE_KEY": redis_queue_key,
            "REDIS_STATUS_KEY": redis_status_key,
            "GENERATED_ROOT": generated_root_value,
            "LTX_MODEL_ID": model_path_value,
            "LTX_DEVICE": device,
        }.items()
        if not value
    ]

    int_keys = [
        "LTX_NUM_FRAMES",
        "LTX_OUTPUT_FPS",
        "LTX_HEIGHT",
        "LTX_WIDTH",
        "LTX_INFERENCE_STEPS",
    ]
    missing.extend(key for key in int_keys if _get(key) is None)

    if missing:
        raise RuntimeError(f"Missing required configuration values: {', '.join(missing)}")

    model_path = Path(model_path_value).expanduser().resolve()
    if not model_path.exists():
        raise RuntimeError(f"Model directory {model_path} does not exist")

    def _int_config(key: str) -> int:
        raw = _get(key)
        try:
            return int(raw)
        except ValueError as exc:
            raise ValueError(f"{key} must be an integer") from exc

    return WorkerConfig(
        redis_url=redis_url,
        redis_queue_key=redis_queue_key,
        redis_status_key=redis_status_key,
        generated_root=Path(generated_root_value).resolve(),
        model_path=model_path,
        device=device,
        frames=_int_config("LTX_NUM_FRAMES"),
        fps=_int_config("LTX_OUTPUT_FPS"),
        height=_int_config("LTX_HEIGHT"),
        width=_int_config("LTX_WIDTH"),
        inference_steps=_int_config("LTX_INFERENCE_STEPS"),
    )


class LTXVideoRunner:
    """Minimal wrapper around the official LTX-Video pipeline."""

    _NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"

    def __init__(self, model_root: Path, device_preference: str = "auto") -> None:
        self.model_root = model_root
        self.config_path = self._discover_config()
        self.device = self._resolve_device(device_preference)

        config = load_pipeline_config(self.config_path)
        self._guidance_scale = config.get("guidance_scale", 1.0)
        self._default_steps = config.get("num_inference_steps", 8)
        self._decode_timestep = config.get("decode_timestep")
        self._decode_noise_scale = config.get("decode_noise_scale")

        checkpoint_name = config["checkpoint_path"]
        checkpoint_path = self.model_root / checkpoint_name
        if not checkpoint_path.exists():
            LOGGER.info("Checkpoint %s missing; downloading from hub", checkpoint_name)
            downloaded = hf_hub_download(
                repo_id="Lightricks/LTX-Video",
                filename=checkpoint_name,
                repo_type="model",
                local_dir=str(self.model_root),
                local_dir_use_symlinks=False,
            )
            checkpoint_path = Path(downloaded)
        LOGGER.info(
            "Loading LTX-Video pipeline (checkpoint %s, config %s)",
            checkpoint_path,
            self.config_path,
        )
        self.pipeline = create_ltx_video_pipeline(
            ckpt_path=str(checkpoint_path),
            precision=config.get("precision", "bfloat16"),
            text_encoder_model_name_or_path=config["text_encoder_model_name_or_path"],
            sampler=config.get("sampler"),
            device=self.device,
            enhance_prompt=False,
        )

    def generate(
        self,
        prompt: str,
        *,
        height: int,
        width: int,
        num_frames: int,
        num_inference_steps: int,
        fps: int,
        output_path: Path,
        progress_callback: Callable[[int], None] | None = None,
    ) -> None:
        """Generate a video for the given prompt and persist it to disk."""
        steps = num_inference_steps or self._default_steps
        total_steps = max(1, steps)
        seed = int(time.time() * 1000) & 0xFFFFFFFF
        seed_everething(seed)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        height_padded = ((height + 31) // 32) * 32
        width_padded = ((width + 31) // 32) * 32
        num_frames_padded = ((max(num_frames, 1) - 2) // 8 + 1) * 8 + 1
        padding = calculate_padding(height, width, height_padded, width_padded)

        last_percent = -1

        def _on_step(_pipe, step: int, _timestep, kwargs: dict) -> dict:
            nonlocal last_percent
            if progress_callback is not None:
                percent = min(99, int(((step + 1) * 100) / total_steps))
                if percent != last_percent:
                    last_percent = percent
                    progress_callback(percent)
            return kwargs

        call_kwargs: dict[str, object] = {
            "prompt": prompt,
            "negative_prompt": self._NEGATIVE_PROMPT,
            "height": height_padded,
            "width": width_padded,
            "num_frames": num_frames_padded,
            "num_inference_steps": steps,
            "frame_rate": fps,
            "output_type": "pt",
            "generator": generator,
            "guidance_scale": self._guidance_scale,
            "vae_per_channel_normalize": True,
            "is_video": True,
        }
        if progress_callback is not None:
            call_kwargs["callback_on_step_end"] = _on_step
        if self._decode_timestep is not None:
            call_kwargs["decode_timestep"] = self._decode_timestep
        if self._decode_noise_scale is not None:
            call_kwargs["decode_noise_scale"] = self._decode_noise_scale

        outputs = self.pipeline(**call_kwargs)
        images = outputs.images

        pad_left, pad_right, pad_top, pad_bottom = padding
        pad_bottom = -pad_bottom or images.shape[3]
        pad_right = -pad_right or images.shape[4]
        images = images[:, :, :num_frames, pad_top:pad_bottom, pad_left:pad_right]

        video_np = images[0].permute(1, 2, 3, 0).cpu().float().numpy()
        video_np = np.clip(video_np, 0.0, 1.0)
        video_np = (video_np * 255).astype(np.uint8)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Writing %s (%d frames @ %dfps)", output_path, video_np.shape[0], fps)
        with imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8) as writer:
            for frame in video_np:
                writer.append_data(frame)

        if progress_callback is not None and last_percent < 99:
            progress_callback(99)

    def _discover_config(self) -> str:
        configs_dir = self.model_root / "configs"
        if configs_dir.is_dir():
            for pattern in ("*.yaml", "*.yml"):
                files = sorted(configs_dir.glob(pattern))
                if files:
                    return str(files[0])

        LOGGER.info("Pipeline config not found locally; installing packaged default")
        packaged = pkg_files("ltx_video.configs").joinpath("ltxv-2b-0.9.6-distilled.yaml")
        target = self.model_root / "configs" / "ltxv-2b-0.9.6-distilled.yaml"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(packaged.read_bytes())
        return str(target)

    @staticmethod
    def _resolve_device(preference: str) -> str:
        if preference and preference != "auto":
            return preference
        return ltx_get_device()



class InferenceWorker:
    """Worker that polls Redis for jobs and executes inference."""

    def __init__(self, config: WorkerConfig):
        self.config = config
        self.redis = Redis.from_url(config.redis_url, decode_responses=True)
        self.status_key = config.redis_status_key
        self.metadata_key = f"{self.status_key}:metadata"
        self.runner = LTXVideoRunner(config.model_path, config.device)

    def run(self) -> None:
        LOGGER.info("Starting inference worker listening on queue '%s'", self.config.redis_queue_key)
        while True:
            try:
                job = self._dequeue_job()
                if job is None:
                    continue
                self._process_job(job)
            except KeyboardInterrupt:
                LOGGER.info("Inference worker interrupted; shutting down.")
                break
            except Exception:
                LOGGER.exception("Unhandled exception in worker loop; continuing in 5 seconds.")
                time.sleep(5)

    def _set_status(self, job_id: str, state: str, percent: int | None = None) -> None:
        self.redis.hset(self.status_key, job_id, _format_status(state, percent))

    def _dequeue_job(self) -> dict | None:
        """Block on the Redis queue for the next job using BLPOP."""
        result = self.redis.blpop(self.config.redis_queue_key, timeout=0)

        _, payload = result
        LOGGER.info("Dequeued job payload: %s", payload)
        try:
            job = json.loads(payload)
        except json.JSONDecodeError as exc:
            LOGGER.error("Invalid job payload; discarding. Error: %s", exc)
            return None

        if "job_id" not in job or "prompt" not in job:
            LOGGER.error("Job payload missing required keys; discarding: %s", job)
            return None
        return job

    def _process_job(self, job: dict) -> None:
        job_id = job["job_id"]
        prompt = job["prompt"]
        handler = job.get("handled_by")
        output_path = self.config.generated_root / job_id / "out.mp4"
        last_percent = 0
        try:
            if handler:
                self.redis.hset(self.metadata_key, job_id, handler)
            self._set_status(job_id, "running", 0)

            def progress_callback(percent: int) -> None:
                nonlocal last_percent
                bounded = max(0, min(99, percent))
                if bounded != last_percent:
                    last_percent = bounded
                    self._set_status(job_id, "running", bounded)

            self.runner.generate(
                prompt,
                num_frames=self.config.frames,
                height=self.config.height,
                width=self.config.width,
                num_inference_steps=self.config.inference_steps,
                fps=self.config.fps,
                output_path=output_path,
                progress_callback=progress_callback,
            )
            self._set_status(job_id, "completed", 100)
            LOGGER.info("Job %s completed successfully.", job_id)
        except Exception:
            self._set_status(job_id, "failed", min(100, last_percent))
            LOGGER.exception("Job %s failed during inference.", job_id)


def main() -> None:
    config = load_config()
    worker = InferenceWorker(config)
    worker.run()


if __name__ == "__main__":
    main()
