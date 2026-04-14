# Dispatch API Reference

Base URL: `https://api.orbit.example.com/v1`

## Endpoints

### POST /jobs
Create a new job. Body: `{ "customer_id": str, "address": str, "window_start": iso8601, "window_end": iso8601 }`. Returns the created job with a generated `job_id`.

### GET /jobs/{job_id}
Fetch a single job by id.

### POST /jobs/{job_id}/assign
Assign a job to a technician. Body: `{ "technician_id": str }`. Emits a `job.assigned` event.

### GET /technicians/{technician_id}/route
Return the technician's current optimized route for the day.

## Authentication

All endpoints require a bearer JWT. Tokens are issued by `POST /auth/login` and expire after 12 hours.
