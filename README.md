# NeuroStudio

**A self-hosted, local-first AI training studio.** Inspired by Google's Teachable Machine,
extended into a general ML workspace: import data, train real models on your own CPU,
watch live metrics, evaluate honestly, and deploy through a local inference API —
no cloud, no accounts, no telemetry.

```
  ◆ NeuroStudio 0.1.0
  Local URL:  http://127.0.0.1:8765
  Data dir:   ~/.local/share/neurostudio
  Telemetry:  disabled
```

## What actually works today (v0.1.0)

NeuroStudio never advertises a capability it cannot run. Here is the honest list:

| Capability | Status |
|---|---|
| Project workspace (create / archive / duplicate / export) | ✅ |
| Dataset studio: upload images (files / ZIP), classes, dedupe, corrupt detection | ✅ |
| Seeded, stratified train/val/test splits with leakage & imbalance warnings | ✅ |
| Synthetic sample dataset generator (shapes) for a zero-setup first run | ✅ |
| **Real image-classification training** — pure-Go CNN, hand-written forward/backward, gradient-checked | ✅ |
| Adam / SGD, LR schedules, dropout, augmentation, early stopping, checkpoints | ✅ |
| Live training dashboard: SSE event stream, per-step progress, loss curves, safe cancel | ✅ |
| Held-out evaluation: accuracy, macro-F1, confusion matrix, per-class metrics, latency p50/p95 | ✅ |
| Experiment tracking (every run recorded with exact config + metrics) | ✅ |
| Model registry: versioned weights, SHA-256 checksums, downloads | ✅ |
| Inference: REST `POST /predict` + in-browser Playground with drag & drop | ✅ |
| SQLite storage with migrations, integrity check, backup/restore | ✅ |
| Remote access mode with token auth (opt-in) | ✅ |
| Object detection / audio / pose / text / tabular training | 🚧 planned (data storage & annotation export already scaffolded) |
| ONNX export, PyTorch/Keras external backends | 🚧 planned — backend interface exists, native engine ships first |
| GPU acceleration | ❌ not supported — CPU only, by design, at this stage |

The shipped training engine (`go-native`) computes every forward pass, gradient and
optimizer step with code in `internal/ml` — no simulated metrics, no random guesses.
It is numerically gradient-checked by unit tests. It is designed for laptop-scale
datasets (tens to a few thousand small images, 16–96 px). It is not a substitute for
PyTorch on large datasets, and it says so in the UI.

## Quick start (60 seconds)

Prerequisites: none for the prebuilt binary (Linux x86-64). To build from source:
[Go 1.22+](https://go.dev/dl/).

```bash
# Option A — prebuilt binary (linux/amd64, in this repository)
./bin/neurostudio

# Option B — build from source
go build -o neurostudio . && ./neurostudio
```

Open **http://127.0.0.1:8765**, then:

1. **New project** → name it.
2. **Generate sample shapes** (Datasets tab) — creates a labeled dataset locally.
3. **Create version** — pins a seeded 70/15/15 split.
4. **Train tab** → pick backend, defaults are fine → **Start training**.
5. Watch the live dashboard; when it finishes, check the evaluation.
6. **Playground tab** → drop in any image → see predictions with latency.

For your own data: upload images (per-class via the class selector, or a ZIP with one
folder per class), then follow the same steps. Full guide: [`docs/SETUP.md`](docs/SETUP.md).

## Why local-first?

- **Your data never leaves the machine.** No telemetry, no update pings, no accounts.
- **Reproducible:** every split, seed, config and metric is stored; runs are comparable.
- **Inspectable:** one binary, one SQLite file, plain files on disk for everything large.

## Architecture in one paragraph

A single Go process serves an embedded frontend (vanilla JS, no build step) and a
versioned REST API under `/api/v1`. Metadata lives in SQLite (migrations, FK constraints,
WAL); images, checkpoints and model weights live on the filesystem under one data
directory. Training backends implement one interface (`internal/backends`) and declare
their capabilities honestly; the built-in engine runs in-process, external workers
(PyTorch/Keras) will run as isolated processes in later stages. Live training updates
stream over SSE; jobs survive restarts (interrupted jobs are marked failed, checkpoints
preserved). See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Repository layout

```
neurostudio/
├── main.go                     # entrypoint, flags, graceful shutdown
├── internal/
│   ├── api/                    # REST API (/api/v1), SSE, middleware, static serving
│   ├── app/                    # config, data-dir layout, health
│   ├── backends/               # Backend interface + registry (+ goruntime engine)
│   ├── datasets/               # import, dedupe, splits, versions, manifests, generator
│   ├── domain/                 # shared types (projects, jobs, models…)
│   ├── ml/                     # tensors, conv/dense kernels, optimizers, metrics
│   ├── projects/               # project service
│   ├── store/                  # SQLite: migrations, backup/restore, integrity
│   ├── training/               # job manager: queue, SSE hub, cancel, experiments
│   ├── util/                   # ids, safe paths, atomic writes, HTTP helpers
│   └── webfs/                  # embedded frontend (html/css/js)
├── docs/                       # SETUP, ARCHITECTURE, API, SECURITY …
├── scripts/smoke.sh            # end-to-end acceptance run against a live server
└── Makefile
```

## Development

```bash
make test     # unit + integration tests (gradient checks, dataset pipeline, E2E training)
make run      # build & start on :8765
make smoke    # boots the binary and walks the full 14-step acceptance workflow
make check    # vet + fmt check
```

The E2E test (`internal/training/manager_test.go`) trains a real CNN on a synthetic
dataset and asserts the evaluation quality — if the engine regresses, tests fail.

## Security posture

Binds to `127.0.0.1` by default; remote access requires `--allow-remote` and prints a
one-time API token. Uploads are size-limited and type-checked; all user-influenced paths
go through a traversal-proof join; nothing uploaded is ever executed. Details and
threat model: [`docs/SECURITY.md`](docs/SECURITY.md).

## License

Apache-2.0 — see [LICENSE](LICENSE). Model weights you train are yours; the license of
your dataset is your responsibility (the registry records it per model).

## Roadmap

Stage 3 (live dashboard, export) is complete in v0.1.0 for the native engine.
Next: ONNX export → PyTorch/Keras external backends → object detection → audio & pose
→ AutoML experiments. See [`docs/ROADMAP.md`](docs/ROADMAP.md).
