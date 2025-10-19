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
    server_ready_value = os.environ.get("SERVER_READY_FILE") or os.environ.get("MODEL_READY_FILE")
    server_building_value = os.environ.get("SERVER_BUILDING_FILE") or os.environ.get("MODEL_DOWNLOADING_FILE")

    if not redis_url or not redis_queue_key or not redis_status_key or not generated_root_value:
        raise RuntimeError(
            "REDIS_URL, REDIS_QUEUE_KEY, REDIS_STATUS_KEY, and GENERATED_ROOT must be set in .env"
        )

    redis_client = Redis.from_url(redis_url, decode_responses=True)
    generated_root = Path(generated_root_value).resolve()
    server_ready_path = Path(server_ready_value).resolve() if server_ready_value else None
    server_building_path = Path(server_building_value).resolve() if server_building_value else None
    app_instance = os.environ.get("APP_INSTANCE") or socket.gethostname()
    job_metadata_key = f"{redis_status_key}:metadata"

    def _format_status(state: str, percent: int | None = None) -> str:
        if percent is None:
            return state
        bounded = max(0, min(100, percent))
        return f"{state}:{bounded}"

    def _server_status() -> dict[str, object]:
        ready = True
        building = False
        message = ""

        if server_ready_path is not None:
            ready = server_ready_path.exists()
            if ready:
                message = "Server ready"
            else:
                building = server_building_path.exists() if server_building_path else False
                message = (
                    "Server is currently building. Generation will be available once preparation completes."
                    if building
                    else "Server is not yet ready."
                )
        else:
            message = "Server status not tracked."

        return {"ready": ready, "building": building, "message": message}

    @app.get("/")
    def index():
        """Serve frontend."""
        return render_template("index.html")

    @app.get("/api/model-status")
    def get_model_status():
        """Report whether the inference server is ready for use."""
        return jsonify(_server_status()), 200

    @app.post("/api/generate")
    def generate():
        """Start video inference job."""
        payload = request.get_json(silent=True) or {}
        prompt = payload.get("prompt")
        if not prompt:
            return jsonify({"error": "prompt is required"}), 400

        status = _server_status()
        if not status.get("ready", False):
            return (
                jsonify({"error": "model_not_ready", "message": status.get("message")}),
                503,
            )

        job_id = str(uuid.uuid4())
        job_descriptor = {"job_id": job_id, "prompt": prompt, "handled_by": app_instance}
        with redis_client.pipeline() as pipe:
            pipe.hset(redis_status_key, job_id, _format_status("queued", 0))
            pipe.hset(job_metadata_key, job_id, app_instance)
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

        handler = app_instance
        response["handled_by"] = handler
        redis_client.hset(job_metadata_key, job_id, handler)
        if percent is not None:
            response["percent_complete"] = percent

        return jsonify(response), 200

    @app.get("/generated/<job_id>/out.mp4")
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
