# Setup Guide

Everything runs locally. No account, no cloud service, no API keys.

## 1. Prerequisites

| Goal | Requirement |
|---|---|
| Run the prebuilt binary | none (Linux x86-64 included; other platforms: build from source) |
| Build from source | [Go 1.22+](https://go.dev/dl/) (any OS: Windows, macOS, Linux) |
| Hardware | any modern laptop; 2+ CPU cores recommended; ~100 MB free disk to start |

There are **no Python, CUDA, Docker or database prerequisites**. SQLite is embedded;
the training engine is compiled in.

## 2. Get NeuroStudio

```bash
unzip neurostudio.zip && cd neurostudio
```

## 3. Run

### Linux / macOS
```bash
./bin/neurostudio            # prebuilt (linux/amd64)
# or build for your platform:
go build -o neurostudio . && ./neurostudio
```

### Windows (PowerShell)
```powershell
go build -o neurostudio.exe .
.\neurostudio.exe
```

You should see:

```
  NeuroStudio 0.1.0
  Local URL:  http://127.0.0.1:8765
  Data dir:   /home/you/.local/share/neurostudio
  Telemetry:  disabled
  Press Ctrl+C to stop.
```

Open **http://127.0.0.1:8765**.

## 4. First run (no data needed)

The fastest way to verify everything works — uses only locally generated data:

1. **Projects → New project** (task: *Image classification*).
2. Open the project → **Datasets → ✨ Generate sample shapes** (e.g. 3 classes × 30).
3. Still in Datasets: **Create version** with defaults (0.70 / 0.15 / 0.15, seed 42).
4. **Train tab** → click the `go-native` backend card → defaults are sensible → **▶ Start training**.
5. The **Jobs** dashboard opens: live epoch progress, loss curves, step ticker.
6. When it completes you get: held-out accuracy, macro-F1, confusion matrix, per-class
   metrics, latency, and honest warnings (overfitting / imbalance) when detected.
7. **Playground tab** → drop any image (even one you screenshot from the dataset grid) →
   predictions + inference latency, computed locally.
8. **Models tab** → download the trained model (a portable JSON with base64 weights
   and SHA-256 recorded).

## 5. Use your own images

Two supported import paths:

- **Files:** create classes first, select a class in the dropdown, then *Import images*.
- **ZIP (recommended):** one top-level folder per class, images inside:
  ```
  mydata.zip
  ├── cats/  img001.jpg ...
  └── dogs/  img001.jpg ...
  ```
  Folder names become classes automatically.

Quality guidance for the native engine (be realistic — it is a small CPU model):

- 30–200+ images per class helps a lot; more is better.
- Input size 48–64 px is a good accuracy/speed trade-off (16–96 supported).
- Watch the dataset warnings after creating a version: duplicates, leakage,
  class imbalance and min-sample checks are surfaced, never hidden.

## 6. Configuration reference

| Flag | Default | Meaning |
|---|---|---|
| `--port` | `8765` | HTTP port |
| `--host` | `127.0.0.1` | bind address (loopback only unless `--allow-remote`) |
| `--data-dir` | OS user-data dir (or `$NEUROSTUDIO_DATA`) | where everything lives |
| `--allow-remote` | off | bind non-loopback + enable token auth (token printed once at startup) |
| `--dev-frontend` | off | serve the SPA from a directory (frontend development) |
| `--verbose` | off | debug logging |

The data directory layout (safe to back up as a whole):

```
<data-dir>/
├── neurostudio.db      # SQLite metadata (WAL)
├── datasets/<id>/      # image files + manifests per dataset version
├── jobs/<jobId>/       # checkpoints, model.native.json, logs
├── models/             # (registry points into jobs/; exports here)
├── exports/            # generated bundles
└── backups/            # database snapshots from the System tab
```

## 7. Remote access (optional)

```bash
./neurostudio --allow-remote --host 0.0.0.0 --port 8765
```

The console prints an API token. The web UI will ask for it once (stored in
localStorage); scripts pass `Authorization: Bearer <token>` (SSE endpoints also accept
`?token=`). Only do this on networks you trust — the API can train jobs and read data.

## 8. Backup & restore

- **UI:** System → *Create backup now* (SQLite snapshot into `backups/`), then
  *Stage restore* (applied on next start while no jobs run).
- **Full backup:** stop the app and copy the whole data directory — that includes
  images and model weights, which the DB snapshot alone does not.

## 9. Upgrading & troubleshooting

- **Upgrade:** replace the binary; the SQLite schema migrates automatically at startup.
- **Port taken:** `--port 9000`.
- **"no held-out samples"** when starting training: create a dataset *version* first
  (the split step) — training never trains on 100% of the data.
- **Training slow:** reduce input size to 24–32 px or the image count for a first
  pass; the native engine parallelizes convolutions across cores.
- **Interrupted jobs:** if the process dies mid-training, the job is marked *failed*
  on next boot and its checkpoints are kept — start a new job from the same version;
  nothing is silently re-run.
- **Reset everything:** stop the app and delete the data directory.

## 10. Verify the installation

```bash
./scripts/smoke.sh     # walks the complete 14-step workflow against a live server
make test              # unit + integration tests
```
