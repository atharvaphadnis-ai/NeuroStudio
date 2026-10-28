# Contributing to NeuroStudio

Thank you for helping build a genuinely local, honest ML studio. A few rules keep this
project trustworthy.

## The honesty rule (non-negotiable)

NeuroStudio **never**:

- fakes metrics, predictions, or training progress;
- advertises a task or backend that has no real implementation behind it;
- hides warnings (data leakage, imbalance, overfitting) to make numbers look better;
- adds cloud calls, telemetry, or third-party services.

Any PR that breaks these will be declined, however good the code is otherwise.

## Ground rules

1. **Real capability per stage.** A feature lands when its end-to-end workflow works
   (import → train → evaluate → use). Partial features stay behind clear "not yet"
   messages instead of half-working UI.
2. **Tests with the code.** Bug fixes need a regression test; new training math needs a
   gradient check or learnability test; new endpoints need a handler test.
3. **One dependency policy.** Keep the dependency tree tiny (currently: SQLite driver
   only). Justify any addition in the PR description.
4. **No build step for the frontend.** Vanilla ES modules in `internal/webfs/files`;
   keep it framework-free.
5. **Compatibility:** support the last two Go releases. No cgo.

## Development workflow

```bash
git clone <your fork>
cd neurostudio
make test      # everything must pass
make run       # try your change at http://127.0.0.1:8765
make smoke     # end-to-end acceptance run
```

- Branch from `main`; keep PRs focused (one feature or fix per PR).
- `go vet ./...` and `gofmt` clean.
- Write commit messages in the imperative: "add early stopping to trainer".
- Update docs (README table, docs/API.md) when behavior changes.

## Adding a training backend

1. Implement `backends.Backend` (`ID`, `Name`, `Discover`, `Train`).
2. `Discover` must report **real** capabilities — probe the environment; never assume.
3. Train must stream `domain.JobEvent`s through `emit` (persisted: phase/metrics/
   checkpoint/log; live-only: tick) and honor context cancellation at batch boundaries.
4. Register it in `main.go`; the UI picks it up automatically.
5. Add a test that runs a tiny real job through the manager.

## Reporting bugs

Open a GitHub issue with: what you did, what you expected, what happened, and the
`--verbose` log excerpt. For training-quality reports, include the dataset *version id*
and the experiment config from the UI — reproducibility is the point.

## Code of Conduct

Be excellent to each other. The short version of [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md):
harassment-free participation is mandatory; maintainers enforce it.
