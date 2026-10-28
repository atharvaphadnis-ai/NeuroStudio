# Roadmap

Statuses are honest: a stage is "done" only when its end-to-end workflow passes the
acceptance checklist on a plain laptop.

## ✅ Stage 1 — Foundation (v0.1.0)
- Go server, embedded SPA, SQLite store with migrations, backup/restore
- Project workspace, runtime diagnostics (RAM/disk/DB integrity)
- Data-directory layout, local-first security posture, remote token mode

## ✅ Stage 2 — Dataset studio + real image-classification training (v0.1.0)
- Import (multipart/ZIP), sha256 dedupe, corrupt detection, classes, relabeling
- Seeded stratified splits with leakage/imbalance warnings; version manifests
- **Native Go training engine**: gradient-checked CNN, Adam/SGD, schedules,
  dropout, augmentation, early stopping, checkpoints, safe cancel
- Live SSE dashboard with per-step progress and loss curves
- Held-out evaluation: accuracy, macro-F1, confusion matrix, per-class, latency
- Experiment tracking, model registry (native-1 format, checksums), REST inference

## 🔜 Stage 3 — Sharpening the loop
- Model export: ONNX (opset-pinned, with conversion notes and limits stated per model)
- Browser-side inference runtime (wasm) for the Playground
- Checkpoint resume ("continue training from epoch N")
- Test-split playground view + per-image explanations (top-k with tensors of attention-free baselines: only honest, computed values)

## 🔜 Stage 4 — External training backends
- Process-isolated Python worker protocol (stdout framing, capability discovery,
  dependency checks with actionable install hints)
- PyTorch / TensorFlow-Keras adapters for transfer learning on larger images
- Honest hardware pages: what a given config can realistically train, and how long

## 🔜 Stage 5 — More tasks (in data-model order, never before their pipeline exists)
- Object detection (bbox annotation editor, YOLO/COCO already exportable)
- Audio classification (recording UI, spectrogram pipeline)
- Pose classification (keypoint capture from webcam)

## 🔜 Stage 6 — Power features
- AutoML-style small experiment sweeps with comparison tables
- Local LLM Studio (optional module, quantized GGUF inference; training stays with
  external backends) — only ever an explicit download with size disclosure
- Plugin manifest format for third-party backends/importers

## 🔜 Stage 7 — Release engineering
- Signed builds for Windows/Linux/macOS, auto-update optional and off by default
- Onboarding tour, sample project gallery
- Accessibility pass (keyboard nav, contrast, screen-reader labels)
