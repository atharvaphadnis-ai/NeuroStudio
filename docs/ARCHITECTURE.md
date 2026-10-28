# Architecture

## Principles

1. **Local-first, single process.** One Go binary serves the API and the embedded UI.
   Default bind: loopback. No external services are contacted, ever.
2. **Honest capabilities.** A task/backend combination is advertised only when a real
   pipeline exists behind it. Missing metrics render as "unavailable"; never fabricated.
3. **Metadata in SQLite, blobs on disk.** Anything large (images, weights, checkpoints)
   is a plain file referenced by path; the database stays small, backuppable and safe.
4. **Reproducibility.** Seeded splits, seeded training, manifests pinning every sample
   and its split; experiments store the exact config that produced each metric.

## Component map

```
┌─────────────────────────── one Go process ───────────────────────────┐
│                                                                      │
│  webfs (embedded SPA)          api/ (REST /api/v1, SSE, middleware)  │
│        │                              │            │                 │
│        └── static ◄───────────────────┘            │                 │
│                                             projects/  datasets/     │
│                                                    │            │     │
│                                                 training/Manager │     │
│                                              queue → worker → Hub│     │
│                                                     │            │     │
│                                              backends.Registry   │     │
│                                                     │            │     │
│                                              goruntime.Backend   │     │
│                                                     │                  │
│                                                   ml/ (tensors, CNN,  │
│                                                        optim, metrics)│
│                                                                     │
│  store/ (SQLite: WAL, migrations, backup)    app/ (config, paths)   │
└──────────────────────────────────────────────────────────────────────┘
```

## Request flow: starting and watching a training job

1. `POST /api/v1/projects/{id}/jobs` — the API validates the request against *reality*:
   project exists, dataset version + manifest exist on disk, the chosen backend
   `Discover()` report actually supports the task. Invalid combos get 422 with a reason.
2. The job row is created (`status=queued`) and pushed to the manager's queue.
3. The single worker pops it, flips to `running`, and calls `Backend.Train(ctx, spec, emit)`
   with a manifest path, the datasets root and a per-job work directory.
4. The backend loads the manifest (`neurostudio/manifest-1`: classes, per-sample paths,
   splits, hashes), decodes images, preprocesses to tensors, and trains for real:
   forward → backward → `BatchStep` per batch, augmentation on the train split only.
5. Events flow back through `emit`: `phase`, per-batch `tick` (live only), per-epoch
   `metrics` (persisted), `checkpoint`, `log`, `error`, `done`. The manager assigns
   sequences, persists selected events to `job_events`, updates `jobs.progress`,
   and fans everything out through the SSE `Hub`.
6. `GET /api/v1/jobs/{id}/events` (SSE) replays persisted events (supports
   `Last-Event-ID`) then streams live until `done`.
7. Cancellation is cooperative: `POST /jobs/{id}/cancel` cancels the context; the
   training loop checks it every batch, checkpoints stay on disk.
8. On success the manager writes an `experiments` row (config + metrics + curves) and
   registers the model: `models` + `model_versions` (format `native-1`, size, SHA-256,
   input spec, classes).

## The ML engine (`internal/ml` + `backends/goruntime`)

- `Tensor` — row-major float32, channel-major images `[N,C,H,W]`.
- `Conv2D` — parallel over (sample, filter); backward with per-worker buffers reduced
  deterministically. `MaxPool2`, `ReLU`, `Dense` — hand-written backward passes.
- `SoftmaxCE` — numerically stable softmax + cross-entropy with mean-over-batch grads.
- `Adam` / `SGD(momentum)` with decoupled weight decay; LR schedules (cosine/step).
- Architecture `ns-cnn-small`: 3×(conv3×3 → ReLU → maxpool2) → dense head.
  ~26k conv params + head; sized for 16–96 px inputs on CPU.
- Correctness: `internal/ml` tests include **numerical gradient checks** (analytic vs
  finite-difference) and an end-to-end learnability test; the training package test
  trains on a synthetic dataset and asserts held-out accuracy.

Serialization: the native model format (`neurostudio/native-1`) is JSON with base64
float32 weight tensors, input spec, class order, dropout and metadata — portable and
diffable. Inference (`POST /models/{id}/versions/{vid}/predict`) reloads weights,
applies the identical preprocessing pipeline, and reports measured latency.

## Data model (SQLite)

`projects → datasets → classes / samples (sha256-deduped, status-ok filtering) /
dataset_versions (seeded splits + manifest file) → jobs → job_events, checkpoints →
experiments → models → model_versions`. Foreign keys with cascading deletes, indexes on
every access path, `schema_migrations` for forward-only migrations, WAL mode,
single-connection pool (serialized writes) — nested queries never run inside an open
iterator (that would deadlock; documented and tested).

Interrupted runs: at startup, jobs still marked `queued/running` are marked `failed`
with a clear reason. Nothing is silently re-run; checkpoints remain usable.

## Frontend

Vanilla ES-module SPA served from the binary (`internal/webfs`), no build step, no CDN.
Hash router: projects list → project workspace (Datasets / Train / Jobs / Experiments /
Models / Playground) → System. The Jobs view subscribes to SSE, draws live loss curves
on `<canvas>`, renders the confusion matrix and per-class tables from real metrics.

## Security model (summary — full doc: SECURITY.md)

Loopback bind by default; opt-in remote mode with bearer token. Path traversal is
blocked in one choke point (`util.SafeJoin`); uploads are size/type limited and stored
 inert (never executed); JSON bodies are strictly decoded. Security headers set; no
telemetry. The inference endpoint accepts only decodable images.

## Extension points (implemented interfaces)

| Interface | Package | Notes |
|---|---|---|
| `backends.Backend` (`ID/Name/Discover/Train`) | `internal/backends` | external PyTorch/Keras workers plug in here as isolated processes |
| Dataset importers | `internal/datasets` | multipart + ZIP today; directory/crawler imports next |
| Model formats | `model_versions.format` | `native-1` today; `onnx` when the exporter lands |

## Known limitations (by design, v0.1.0)

- One training job at a time (CPU-bound engine; also a correctness guarantee).
- Native engine: image classification, ≤ 96 px inputs, ≤ 4000 samples/job (1 GiB tensor guard).
- No GPU. Browser-side inference and ONNX export are on the roadmap, not shipped.
