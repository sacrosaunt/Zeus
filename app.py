from flask import Flask, jsonify, request


def create_app() -> Flask:
    app = Flask(__name__)

    @app.post("/api/generate")
    def generate():
        """Start video inference job."""
        payload = request.get_json(silent=True) or {}
        return jsonify({"status": "not_implemented", "payload": payload}), 501

    @app.get("/api/jobs/<job_id>")
    def get_job(job_id: str):
        """Retrieve job status."""
        return jsonify({"status": "not_implemented", "job_id": job_id}), 501

    @app.get("/files/<job_id>/out.mp4")
    def get_job_output(job_id: str):
        """Return generated video file."""
        return jsonify({"status": "not_implemented", "job_id": job_id}), 501

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
