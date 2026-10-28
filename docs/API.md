# HTTP API (v1)

Base: `http://127.0.0.1:8765/api/v1`. All responses are JSON. Errors use the envelope:

```json
{ "status": 404, "code": "not_found", "message": "dataset not found" }
```

Remote mode requires `Authorization: Bearer <token>` on every route except `/` (SSE
also accepts `?token=`). Destructive deletes require `?confirm=true`.

## System
| Method | Path | Description |
|---|---|---|
| GET | `/system/info` | version, platform, data dir, uptime, telemetry statement |
| GET | `/system/diagnostics` | hardware + integrity checks |
| GET | `/system/storage` | byte usage per category + disk free |
| POST | `/system/backup` | write SQLite snapshot → `{file, size}` |
| POST | `/system/restore` | `{file}` — stage restore (applied next start, blocked while jobs run) |

## Projects
| Method | Path | Description |
|---|---|---|
| GET | `/projects?query=&status=` | list (status: `active`/`archived`) |
| POST | `/projects` | `{name, description?, taskType}` |
| GET | `/projects/{id}` | project + counts |
| PATCH | `/projects/{id}` | partial update |
| DELETE | `/projects/{id}?confirm=true` | delete with all data |
| POST | `/projects/{id}/duplicate` · `/archive` · `/restore` | lifecycle |
| GET | `/projects/{id}/export` | project bundle |

## Datasets & samples
| Method | Path | Description |
|---|---|---|
| GET/POST | `/projects/{id}/datasets` | list / create (`{name, dataType, notes?}`) |
| POST | `/projects/{id}/sample-dataset` | `{name?, classes, perClass, seed}` synthetic shapes |
| GET | `/datasets/{id}` | dataset + classes + counts |
| DELETE | `/datasets/{id}?confirm=true` | delete |
| GET | `/datasets/{id}/stats` | per-status/class counts, sizes, dimensions |
| POST | `/datasets/{id}/classes` | `{name}` — add class |
| PATCH/DELETE | `/classes/{id}` | rename / delete |
| POST | `/datasets/{id}/samples/import` | multipart; fields: any file field(s), optional `class`, or `files.<Class>` names for auto-classing |
| POST | `/datasets/{id}/samples/import-zip` | raw ZIP body; top-level folder per class |
| GET | `/datasets/{id}/samples?classId=&status=&page=&pageSize=` | paged samples |
| PATCH | `/samples/{id}` | `{classId: string|null}` relabel |
| POST | `/samples/delete` | `{ids: [...]}` |
| GET | `/samples/{id}/file?size=128` | original file or PNG thumbnail |

## Versions & splits
| Method | Path | Description |
|---|---|---|
| GET/POST | `/datasets/{id}/versions` | list / create `{train, val, test, seed}` (fractions sum to 1) |
| GET | `/datasets/{id}/export/manifest?versionId=` | pinned manifest (classes, samples, splits, hashes) |
| GET | `/datasets/{id}/export/annotations?format=coco|yolo` | annotation export |

## Training
| Method | Path | Description |
|---|---|---|
| GET | `/training/backends` | discovery reports with per-task support + reasons |
| POST | `/projects/{id}/jobs` | create job (see below) |
| GET | `/projects/{id}/jobs` | list, newest first |
| GET | `/jobs/{id}` | job incl. `progress` (phase, epoch, losses, curves, ETA) |
| POST | `/jobs/{id}/cancel` | cooperative cancel at next batch boundary (idempotent) |
| GET | `/jobs/{id}/events` | **SSE**: replays persisted events, then live; `Last-Event-ID` resumable; terminal event `done` |
| GET | `/jobs/{id}/checkpoints` | saved checkpoints + metrics |
| GET | `/jobs/{id}/evaluation` | final held-out evaluation (409 until completed) |

Create-job body:
```json
{
  "name": "run-1",
  "datasetId": "…", "versionId": "…",
  "backendId": "go-native",
  "config": {
    "taskType": "image_classification", "architecture": "ns-cnn-small",
    "input": { "size": 48, "grayscale": false },
    "epochs": 30, "batchSize": 16, "learningRate": 0.003,
    "optimizer": "adam", "weightDecay": 0.0001, "lrSchedule": "cosine",
    "seed": 42, "workers": 2, "dropout": 0.1,
    "earlyStopping": { "enabled": true, "patience": 5, "minDelta": 0.0001 },
    "augmentation": { "hFlip": true, "brightness": true },
    "checkpointEveryEpochs": 5
  }
}
```
Validation is honest: unknown backend → 400; backend lacking the task → 422 with the
reason; missing manifest → 409; impossible config → 400.

SSE event types: `phase`, `metrics` (per epoch), `tick` (per step, live only),
`checkpoint`, `log`, `error`, `status`, `done`. Example:
```
event: metrics
data: {"jobId":"…","seq":12,"type":"metrics","time":"…","payload":{"epoch":3,"epochs":30,"trainLoss":0.42,"valLoss":0.51,"valAcc":0.75,"lr":0.003}}
```

## Experiments & models
| Method | Path | Description |
|---|---|---|
| GET | `/projects/{id}/experiments` | run history with metrics + curves |
| GET/DELETE | `/experiments/{id}` | detail / remove from log |
| GET | `/projects/{id}/models` | registry incl. versions |
| GET/PATCH/DELETE | `/models/{id}` | detail / rename+notes / delete files+rows |
| GET | `/models/{id}/versions` · `/models/{id}/versions/{vid}` | version list / detail |
| GET | `/models/{id}/versions/{vid}/download` | model file (`Content-Disposition`) |

## Inference
| Method | Path | Description |
|---|---|---|
| POST | `/models/{id}/versions/{vid}/predict` | multipart field `image` → predictions |
| POST | `/models/{id}/predict` | same, latest version |

Response:
```json
{ "predictions": [{"label":"circle","score":0.97},{"label":"square","score":0.03}],
  "latencyMs": 1.54, "runtime": "go-native", "modelVersionId": "…" }
```
Predictions are computed locally with the same preprocessing as training. The endpoint
rejects non-image payloads; model files that vanished from disk yield `410 artifact_missing`.
