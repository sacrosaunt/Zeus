import json
import os
import socket
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file
from redis import Redis


def create_app() -> Flask:
    app = Flask(__name__)

    redis_url = os.environ.get("REDIS_URL")
    redis_queue_key = os.environ.get("REDIS_QUEUE_KEY")
    redis_status_key = os.environ.get("REDIS_STATUS_KEY")
    generated_root_value = os.environ.get("GENERATED_ROOT")
    model_ready_value = os.environ.get("MODEL_READY_FILE")
    model_downloading_value = os.environ.get("MODEL_DOWNLOADING_FILE")

    if not redis_url or not redis_queue_key or not redis_status_key or not generated_root_value:
        raise RuntimeError(
            "REDIS_URL, REDIS_QUEUE_KEY, REDIS_STATUS_KEY, and GENERATED_ROOT must be set in .env"
        )

    redis_client = Redis.from_url(redis_url, decode_responses=True)
    generated_root = Path(generated_root_value).resolve()
    model_ready_path = Path(model_ready_value).resolve() if model_ready_value else None
    model_downloading_path = Path(model_downloading_value).resolve() if model_downloading_value else None
    app_instance = os.environ.get("APP_INSTANCE") or socket.gethostname()

    def _format_status(state: str, percent: int | None = None) -> str:
        if percent is None:
            return state
        bounded = max(0, min(100, percent))
        return f"{state}:{bounded}"

    def _model_status() -> dict[str, object]:
        ready = True
        downloading = False
        message = ""

        if model_ready_path is not None:
            ready = model_ready_path.exists()
            if ready:
                message = "Model ready"
            else:
                downloading = model_downloading_path.exists() if model_downloading_path else False
                message = (
                    "Please wait for the model to finish downloading to the application servers. Generation will be available soon."
                    if downloading
                    else "Model is not yet available."
                )
        else:
            message = "Model status not tracked."

        return {"ready": ready, "downloading": downloading, "message": message}

    @app.get("/")
    def index():
        """Serve frontend."""
        return render_template("index.html")

    @app.get("/api/model-status")
    def get_model_status():
        """Report whether the inference model is ready for use."""
        return jsonify(_model_status()), 200

    @app.post("/api/generate")
    def generate():
        """Start video inference job."""
        payload = request.get_json(silent=True) or {}
        prompt = payload.get("prompt")
        if not prompt:
            return jsonify({"error": "prompt is required"}), 400

        status = _model_status()
        if not status.get("ready", False):
            return (
                jsonify({"error": "model_not_ready", "message": status.get("message")}),
                503,
            )

        job_id = str(uuid.uuid4())
        job_descriptor = {"job_id": job_id, "prompt": prompt}
        with redis_client.pipeline() as pipe:
            pipe.hset(redis_status_key, job_id, _format_status("queued", 0))
            pipe.rpush(redis_queue_key, json.dumps(job_descriptor))
            pipe.execute()

        return (
            jsonify(
                {
                    "job_id": job_id,
                    "status": "queued",
                    "percent_complete": 0,
                    "handled_by": app_instance,
                }
            ),
            202,
        )

    @app.get("/api/jobs/<job_id>")
    def get_job_status(job_id: str):
        """Retrieve job status."""
        status_raw = redis_client.hget(redis_status_key, job_id)
        if status_raw is None:
            return jsonify({"error": "job_not_found", "job_id": job_id}), 404

        status = status_raw
        percent = None
        if ":" in status_raw:
            state, maybe_percent = status_raw.split(":", 1)
            status = state
            try:
                percent = int(maybe_percent)
            except ValueError:
                percent = None

        response = {"job_id": job_id, "status": status}
        if percent is not None:
            response["percent_complete"] = percent

        return jsonify(response), 200

    @app.get("/files/<job_id>/out.mp4")
    def get_job_output(job_id: str):
        """Return generated video file."""
        # prevent traversal
        if any(sep in job_id for sep in ("/", "\\")):
            return jsonify({"error": "invalid_job_id"}), 400

        file_path = (generated_root / job_id / "out.mp4").resolve()
        try:
            file_path.relative_to(generated_root)
        except ValueError:
            return jsonify({"error": "invalid_job_id"}), 400

        if not file_path.exists():
            return jsonify({"error": "file_not_found", "job_id": job_id}), 404

        return send_file(file_path, mimetype="video/mp4", as_attachment=False)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
