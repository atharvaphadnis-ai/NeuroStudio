# pyworker (reserved)

This directory will host the external Python training workers (PyTorch /
TensorFlow-Keras adapters) that plug into the `backends.Backend` interface as
process-isolated workers. **Nothing is implemented yet** — the native Go engine
(`internal/backends/goruntime`) is the only shipped backend, and the UI honestly
reports other tasks/backends as unavailable until this lands (see docs/ROADMAP.md,
Stage 4).
