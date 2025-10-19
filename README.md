# Zeus

Zeus provides a Docker-based deployment that serves a web UI for generating videos with the `ltxv-2b-0.9.6-distilled` weights from the `Lightricks/LTX-Video` repository. The stack includes a Flask frontend behind Caddy, a Redis-backed job queue, and a GPU-powered inference worker that pulls prompts from the queue.

## Prerequisites
- Python 3.9+ (to run `deploy.py`)
- Docker and Docker Compose v2
- NVIDIA GPU drivers and NVIDIA Container Toolkit on the host (the inference service defaults to `LTX_DEVICE=cuda`)
- Outbound network access for the initial model download

## Quick Start
- Clone this repository onto the target machine and move into the project root.
- Run `python deploy.py`. The helper script builds all containers, starts Redis, the web server, and the inference worker, and downloads the `ltxv-2b-0.9.6-distilled` weights and matching pipeline config into `models/ltxv-2b-0.9.6-distilled/` from `Lightricks/LTX-Video` if they are missing. The model sync may take a while depending on internet speed.
- Monitor the console output. Once you see “Deployment complete.” the stack is up. You can confirm with `docker compose ps`.
- Stream container logs with `docker compose logs -f`.


## Architecture
- Caddy terminates HTTP on port 80 and load balances requests (least connections) across three identical Flask containers (`app1`, `app2`, `app3`).
- The Flask apps enqueue prompts into Redis, which acts as the central job queue and status store.
- A GPU-enabled inference worker consumes jobs from Redis, loads the official `ltx_video` pipeline with the `ltxv-2b-0.9.6-distilled` checkpoint, and writes outputs to the shared `generated/` volume that the apps serve back to users.
- Persistent assets such as model weights live under `models/`, mounted read-only into the app containers and read/write into the inference container.
- The diagram below illustrates the components and their relationships:

<img src="Zeus%20System%20Design%20Cropped.png" alt="Zeus system architecture diagram" width="640" />

## Using The App
- Open a browser to `http://<host-ip>/` (replace `<host-ip>` with the VM address). The Caddy proxy fans out requests across the Flask instances.
- The frontend becomes available even before the entire app has finished building, but will not allow you to submit inference requests until build is complete.
- Submit a prompt from the UI. Jobs are queued in Redis until the inference container is ready.
- Generated videos are written to `generated/<job-id>/out.mp4`; these files are exposed through the `/generated/<job_id>/out.mp4` endpoint if you need direct downloads.
- The frontend automatically retrieves the generated video and plays it. Past generations are also available for replay and download.

## Managing The Deployment
- `docker compose logs -f inference` tails the inference worker if you need to diagnose GPU or model issues.
- `docker compose down` stops and removes the containers while keeping the `models/` and `generated/` volumes on disk for reuse.

## Performance Optimizations
- Caddy uses `least_conn` load balancing, directing incoming requests to the Flask replica with the fewest active connections to maintain stable latency under load.
- Redis backs the queue and status hashes; its in-memory design delivers sub-millisecond writes/reads, and the simple KV store fit the project's lightweight data model.
- Potential future optimizations:
    - Store generated videos in a database or object storage service for faster retrieval, easy replication, and more complex operations.
    - Add a TTL for generated video files to save storage
- Considered partitioning the GPU (MIG) to run multiple generations simultaneously, but ruled it out: splitting the single GPU would throttle per-job compute and produce longer average inference times, which runs counter to the goals of a fast, demo-friendly deployment.
