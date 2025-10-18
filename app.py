import json
import uuid
from pathlib import Path

from dotenv import dotenv_values
from flask import Flask, jsonify, request, send_file
from redis import Redis


def create_app() -> Flask:
    app = Flask(__name__)
    config = dotenv_values(".env")

    redis_url = config.get("REDIS_URL")
    redis_queue_key = config.get("REDIS_QUEUE_KEY")
    generated_root_value = config.get("GENERATED_ROOT")

    if not redis_url or not redis_queue_key or not generated_root_value:
        raise RuntimeError(
            "REDIS_URL, REDIS_QUEUE_KEY, and GENERATED_ROOT must be set in .env"
        )

    redis_client = Redis.from_url(redis_url, decode_responses=True)
    generated_root = Path(generated_root_value).resolve()

    @app.post("/api/generate")
    def generate():
        """Start video inference job."""
        payload = request.get_json(silent=True) or {}
        prompt = payload.get("prompt")
        if not prompt:
            return jsonify({"error": "prompt is required"}), 400

        job_id = str(uuid.uuid4())
        job_descriptor = {"job_id": job_id, "prompt": prompt}

        try:
            redis_client.rpush(redis_queue_key, json.dumps(job_descriptor))
        except Exception as exc:  # pragma: no cover - network failure guard
            return (
                jsonify(
                    {
                        "error": "failed_to_queue_job",
                        "job_id": job_id,
                        "details": str(exc),
                    }
                ),
                503,
            )

        return jsonify({"job_id": job_id, "status": "queued"}), 202

    @app.get("/api/jobs/<job_id>")
    def get_job(job_id: str):
        """Retrieve job status."""
        return jsonify({"status": "not_implemented", "job_id": job_id}), 501

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
