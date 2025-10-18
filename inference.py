import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
from dotenv import dotenv_values
from redis import Redis

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@dataclass
class WorkerConfig:
    redis_url: str
    redis_queue_key: str
    redis_status_key: str
    generated_root: Path
    model_id: str
    device: str
    frames: int
    fps: int
    height: int
    width: int
    inference_steps: int


def load_config() -> WorkerConfig:
    """Load worker configuration from the .env file."""
    config = dotenv_values(".env")
    redis_url = config.get("REDIS_URL")
    redis_queue_key = config.get("REDIS_QUEUE_KEY")
    redis_status_key = config.get("REDIS_STATUS_KEY")
    generated_root_value = config.get("GENERATED_ROOT")
    model_id = config.get("LTX_MODEL_ID")
    device = config.get("LTX_DEVICE")

    missing = [
        key
        for key, value in {
            "REDIS_URL": redis_url,
            "REDIS_QUEUE_KEY": redis_queue_key,
            "REDIS_STATUS_KEY": redis_status_key,
            "GENERATED_ROOT": generated_root_value,
            "LTX_MODEL_ID": model_id,
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
    missing.extend(key for key in int_keys if config.get(key) is None)

    if missing:
        raise RuntimeError(f"Missing required configuration values: {', '.join(missing)}")

    def _int_config(key: str) -> int:
        raw = config.get(key)
        try:
            return int(raw)
        except ValueError as exc:
            raise ValueError(f"{key} must be an integer") from exc

    return WorkerConfig(
        redis_url=redis_url,
        redis_queue_key=redis_queue_key,
        redis_status_key=redis_status_key,
        generated_root=Path(generated_root_value).resolve(),
        model_id=model_id,
        device=device,
        frames=_int_config("LTX_NUM_FRAMES"),
        fps=_int_config("LTX_OUTPUT_FPS"),
        height=_int_config("LTX_HEIGHT"),
        width=_int_config("LTX_WIDTH"),
        inference_steps=_int_config("LTX_INFERENCE_STEPS"),
    )


class LTXVideoModel:
    """Thin wrapper around the LTX-Video diffusion pipeline."""

    def __init__(self, model_id: str, device_preference: str = "auto"):
        import torch
        from diffusers import AutoPipelineForText2Video

        if device_preference == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = device_preference

        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        LOGGER.info("Loading LTX-Video model %s on %s", model_id, device)
        self.pipeline = AutoPipelineForText2Video.from_pretrained(
            model_id,
            torch_dtype=dtype,
        )
        self.pipeline = self.pipeline.to(device)
        self.device = device

    def generate_frames(
        self,
        prompt: str,
        *,
        num_frames: int,
        height: int,
        width: int,
        num_inference_steps: int,
    ) -> Sequence:
        """Run the model and return a sequence of frames."""
        LOGGER.info(
            "Running inference for prompt '%s' (%d frames, %dx%d, %d steps)",
            prompt,
            num_frames,
            width,
            height,
            num_inference_steps,
        )
        output = self.pipeline(
            prompt=prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
        )
        return output.frames


def frames_to_video(frames: Sequence, output_path: Path, fps: int) -> None:
    """Persist a sequence of PIL frames to an mp4 file."""
    import imageio.v2 as imageio

    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays: List[np.ndarray] = [np.asarray(frame) for frame in frames]
    LOGGER.info("Writing %d frames to %s (fps=%d)", len(arrays), output_path, fps)
    imageio.mimwrite(output_path, arrays, fps=fps, codec="libx264", quality=8)


class InferenceWorker:
    """Worker that polls Redis for jobs and executes inference."""

    def __init__(self, config: WorkerConfig):
        self.config = config
        self.redis = Redis.from_url(config.redis_url, decode_responses=True)
        self.status_key = config.redis_status_key
        self.model = LTXVideoModel(config.model_id, config.device)

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
            except Exception:  # pragma: no cover - runtime guard
                LOGGER.exception("Unhandled exception in worker loop; continuing in 5 seconds.")
                time.sleep(5)

    def _set_status(self, job_id: str, status: str) -> None:
        self.redis.hset(self.status_key, job_id, status)

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
        output_path = self.config.generated_root / job_id / "out.mp4"
        try:
            self._set_status(job_id, "running")
            frames = self.model.generate_frames(
                prompt,
                num_frames=self.config.frames,
                height=self.config.height,
                width=self.config.width,
                num_inference_steps=self.config.inference_steps,
            )
            frames_to_video(frames, output_path, fps=self.config.fps)
            self._set_status(job_id, "completed")
            LOGGER.info("Job %s completed successfully.", job_id)
        except Exception as exc:
            self._set_status(job_id, "failed")
            LOGGER.exception("Job %s failed during inference.", job_id)


def main() -> None:
    config = load_config()
    worker = InferenceWorker(config)
    worker.run()


if __name__ == "__main__":
    main()
