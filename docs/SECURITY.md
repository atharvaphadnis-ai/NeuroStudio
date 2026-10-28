# Security Policy

NeuroStudio is a local-first application. This document describes the security model,
the guarantees, and how to report vulnerabilities.

## Threat model & posture

**What NeuroStudio protects:** your machine from malicious inputs, and your data from
leaving it.

- **No telemetry, no phones home.** The binary performs zero outbound network requests.
  There is no update check, no analytics, no crash reporter.
- **Loopback by default.** The server binds `127.0.0.1`. Remote binding requires the
  explicit `--allow-remote` flag, which also generates a fresh API token printed once.
- **Uploads are inert.** Images are decoded with Go's image package; a malformed or
  hostile file fails decoding and is marked `corrupt`. Nothing uploaded is ever
  executed, extracted as code, or passed to a shell.
- **Path traversal is blocked at a single choke point** (`util.SafeJoin`): absolute
  paths, `..` segments, and NUL bytes are rejected; the resolved path must stay inside
  the data directory. All user-influenced paths flow through it.
- **Bounded resources.** Upload size limit (300 MB total body), per-file size checks,
  sample-count and tensor-memory guards in the trainer, single-connection SQLite with
  WAL, request read timeouts, slow-consumer-safe SSE (drop, don't block training).
- **Strict JSON** decoding rejects unknown fields on all API inputs.
- **Security headers:** `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`,
  `Referrer-Policy: no-referrer`. The API and static files are same-origin only; the UI
  performs no cross-origin requests.

## Remote mode

`--allow-remote` is for trusted networks. All API routes (except the SPA shell) require
`Authorization: Bearer <token>`; the token is random 256-bit, generated per launch,
shown once in the console. SSE supports `?token=` because EventSource cannot set
headers. There is no user database by design — one token, one machine, one user.

## Data handling

- Everything (projects, datasets, models) stays in the data directory you configure.
- Deleting a dataset/model removes its files; a model file that is missing yields
  `410 artifact_missing` rather than silent failure.
- Database restores are staged and applied at startup only — never swapped under a
  running job.

## Known limitations (honest list)

- The remote token lives in process memory and is printed to the console; a host-level
  attacker with console read access can read it.
- No rate limiting on local API (the local user already has full machine access).
- Checkpoint and model files are JSON; importing model files from third parties is not
  executed, but standard caution applies to any downloaded content.

## Reporting a vulnerability

Please use GitHub **private security advisories** (Security tab → Report a
vulnerability) or open a security issue marked CONFIDENTIAL. Do not disclose publicly
until a fix is released. Target response time: 72 hours for triage, 14 days for a fix
on critical issues. Supported versions: the latest release.
