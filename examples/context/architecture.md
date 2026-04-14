# Architecture

Orbit is split into three services:

## Dispatch API
A FastAPI service that owns job assignment and publishes events to Kafka. It reads and writes to a Postgres database called `orbit_core`.

## Routing Worker
A Python worker that consumes `job.assigned` events from Kafka and computes optimal routes using OSRM. Results are cached in Redis for 15 minutes.

## Mobile Gateway
A Node.js BFF that fronts the mobile apps. It authenticates technicians via JWTs issued by the Dispatch API and forwards calls over gRPC.

Data flows from Dispatch API → Kafka → Routing Worker, then back to Dispatch API via a `route.computed` topic. The Mobile Gateway polls Dispatch API for the latest state.
