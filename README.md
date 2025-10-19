# Zeus

Zeus provides a Docker-based deployment that serves a web UI for generating videos with the LTX-Video model. The stack includes a Flask frontend behind Caddy, a Redis-backed job queue, and a GPU-powered inference worker that pulls prompts from the queue.

## Prerequisites
- Python 3.9+ (to run `deploy.py`)
- Docker and Docker Compose v2
- NVIDIA GPU drivers and NVIDIA Container Toolkit on the host (the inference service defaults to `LTX_DEVICE=cuda`)
- Outbound network access for the initial model download

## Quick Start
- Clone this repository onto the target machine and move into the project root.
- Run `python deploy.py`. The helper script builds all containers, starts Redis, the web server, and the inference worker, and downloads the LTX-Video weights into `models/LTX-Video/` if they are missing. The model sync may take a while depending on internet speed.
- Monitor the console output. Once you see “Deployment complete.” the stack is up. You can confirm with `docker compose ps`.
- Stream container logs with `docker compose logs -f`.

## Using The App
- Open a browser to `http://<host-ip>/` (replace `<host-ip>` with the VM address). The Caddy proxy fans out requests across the Flask instances.
- The frontend becomes available even before the entire app has finished building, but will not allow you to submit inference requests until build is complete.
- Submit a prompt from the UI. Jobs are queued in Redis until the inference container is ready.
- Generated videos are written to `generated/<job-id>/out.mp4`; these files are exposed through the `/files/<job_id>/out.mp4` endpoint if you need direct downloads.

## Managing The Deployment
- `docker compose logs -f inference` tails the inference worker if you need to diagnose GPU or model issues.
- `docker compose down` stops and removes the containers while keeping the `models/` and `generated/` volumes on disk for reuse.\
