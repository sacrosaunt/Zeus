# Zeus API Reference

The Zeus web server exposes a REST-style API for submitting video generation jobs, tracking progress, and retrieving outputs. All endpoints are served from the same host as the frontend (`http://<host-ip>/`).

## Authentication
The API is un-authenticated. Ensure you only expose the deployment to trusted users or place it behind your own auth proxy.

## Endpoints

### GET `/api/model-status`
Returns the readiness of the inference worker.

**Response 200**
```json
{
  "ready": true,
  "building": false,
  "message": "Server ready"
}
```

When the worker is still syncing model weights you will see `"ready": false` and `"building": true`. Once preparation finishes the `"ready"` flag flips to `true`.

### POST `/api/generate`
Queues a new generation job.

**Request Headers**
- `Content-Type: application/json`

**Request Body**
```json
{
  "prompt": "Describe the video you want"
}
```

**Responses**
- `202 Accepted` – job created successfully.

  ```json
  {
    "job_id": "8b4fbf9e-1e03-4e74-ae43-3ba0f6428b1e",
    "status": "queued",
    "percent_complete": 0,
    "handled_by": "app1"
  }
  ```

- The `handled_by` field echoes which Flask container served this particular response. This is useful for demonstrating Caddy `least_conn` load balancing across replicas.
- `400 Bad Request` – missing or empty prompt.
- `503 Service Unavailable` – model not ready. Response includes an error code and human-readable message.

  ```json
  {
    "error": "model_not_ready",
    "message": "Server is currently building. Generation will be available once preparation completes."
  }
  ```

### GET `/api/jobs/<job_id>`
Retrieves the latest status for a previously submitted job.

**Path Parameter**
- `<job_id>` – the UUID returned when the job was created.

**Responses**
- `200 OK`

  ```json
  {
    "job_id": "8b4fbf9e-1e03-4e74-ae43-3ba0f6428b1e",
    "status": "running",
    "percent_complete": 63,
    "handled_by": "app3"
  }
  ```

- `handled_by` reflects the container name that answered this poll request. Depending on the load-balancing decision, it may change between calls.
- `404 Not Found` – unknown job id.

### GET `/files/<job_id>/out.mp4`
Streams the generated video file.

**Path Parameter**
- `<job_id>` – same UUID as above. Only alphanumeric IDs without path separators are accepted.

**Responses**
- `200 OK` – returns an `video/mp4` stream.
- `400 Bad Request` – malformed job id (e.g., contains slashes).
- `404 Not Found` – no output available for that job.

## Job Lifecycle
1. Submit a prompt to `/api/generate`.
2. Poll `/api/jobs/<job_id>` until `status` becomes `finished`.
3. When finished, download the rendered MP4 from `/files/<job_id>/out.mp4`.

## Error Handling
- Any unexpected server issue returns a 500 with a JSON body containing `"error"` and `"message"`.
- When the inference worker is offline, `/api/model-status` will report `"ready": false`. Clients should refrain from calling `/api/generate` until the model is ready.
- Interpret readiness flags as follows: `"ready": false, "building": true` means deployment is still preparing (keep polling), whereas `"ready": false, "building": false` indicates a fault condition or shutdown that likely requires some sort of intervention.
