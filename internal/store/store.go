// Package store owns the SQLite database: connection lifecycle, schema
// migrations, integrity checks, and backup/restore. Metadata lives in
// SQLite; large binary artifacts (images, weights, checkpoints) always live
// on the filesystem and are referenced by path.
package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	_ "modernc.org/sqlite"
)

type Store struct {
	db   *sql.DB
	path string
	// modernc.org/sqlite allows a single writer; we serialize writes to
	// avoid SQLITE_BUSY under concurrent goroutines.
	mu sync.Mutex
}

// Open opens (creating if needed) the database at path and applies
// migrations.
func Open(path string) (*Store, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, fmt.Errorf("create data dir: %w", err)
	}
	dsn := "file:" + path +
		"?_pragma=journal_mode(WAL)" +
		"&_pragma=busy_timeout(10000)" +
		"&_pragma=foreign_keys(1)" +
		"&_pragma=synchronous(NORMAL)"
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, err
	}
	// Single connection: correctness and simplicity over read parallelism
	// for a local single-user application.
	db.SetMaxOpenConns(1)
	db.SetConnMaxLifetime(0)
	s := &Store{db: db, path: path}
	if err := s.migrate(); err != nil {
		db.Close()
		return nil, err
	}
	return s, nil
}

func (s *Store) Close() error { return s.db.Close() }

func (s *Store) Path() string { return s.path }

// DB exposes the raw handle for packages that need it (kept internal to the
// module via wrapper methods elsewhere).
func (s *Store) DB() *sql.DB { return s.db }

// Lock serializes write transactions.
func (s *Store) Lock() func() {
	s.mu.Lock()
	return s.mu.Unlock
}

type migration struct {
	id  int
	sql string
}

var migrations = []migration{
	{1, `
CREATE TABLE settings (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE projects (
  id          TEXT PRIMARY KEY,
  name        TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  task_type   TEXT NOT NULL DEFAULT 'image_classification',
  status      TEXT NOT NULL DEFAULT 'active',
  settings    TEXT NOT NULL DEFAULT '{}',
  created_at  TEXT NOT NULL,
  updated_at  TEXT NOT NULL,
  archived_at TEXT
);
CREATE INDEX idx_projects_status ON projects(status);
CREATE TABLE datasets (
  id         TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  name       TEXT NOT NULL,
  data_type  TEXT NOT NULL,
  notes      TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX idx_datasets_project ON datasets(project_id);
CREATE TABLE classes (
  id         TEXT PRIMARY KEY,
  dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  name       TEXT NOT NULL,
  color      TEXT NOT NULL DEFAULT '',
  ord        INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  UNIQUE(dataset_id, name)
);
CREATE TABLE samples (
  id          TEXT PRIMARY KEY,
  dataset_id  TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  class_id    TEXT REFERENCES classes(id) ON DELETE SET NULL,
  filename    TEXT NOT NULL,
  rel_path    TEXT NOT NULL,
  media_type  TEXT NOT NULL,
  size_bytes  INTEGER NOT NULL,
  sha256      TEXT NOT NULL,
  width       INTEGER NOT NULL DEFAULT 0,
  height      INTEGER NOT NULL DEFAULT 0,
  duration_ms INTEGER NOT NULL DEFAULT 0,
  status      TEXT NOT NULL DEFAULT 'ok',
  meta        TEXT NOT NULL DEFAULT '{}',
  created_at  TEXT NOT NULL
);
CREATE INDEX idx_samples_dataset ON samples(dataset_id);
CREATE INDEX idx_samples_class   ON samples(class_id);
CREATE INDEX idx_samples_sha     ON samples(dataset_id, sha256);
CREATE TABLE dataset_versions (
  id            TEXT PRIMARY KEY,
  dataset_id    TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  version       INTEGER NOT NULL,
  seed          INTEGER NOT NULL,
  params        TEXT NOT NULL,
  splits        TEXT NOT NULL,
  manifest_path TEXT NOT NULL,
  stats         TEXT NOT NULL,
  created_at    TEXT NOT NULL,
  UNIQUE(dataset_id, version)
);
CREATE TABLE annotations (
  id         TEXT PRIMARY KEY,
  sample_id  TEXT NOT NULL REFERENCES samples(id) ON DELETE CASCADE,
  kind       TEXT NOT NULL,
  label      TEXT NOT NULL DEFAULT '',
  data       TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX idx_annotations_sample ON annotations(sample_id);
CREATE TABLE jobs (
  id                 TEXT PRIMARY KEY,
  project_id         TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  dataset_id         TEXT REFERENCES datasets(id) ON DELETE SET NULL,
  dataset_version_id TEXT REFERENCES dataset_versions(id) ON DELETE SET NULL,
  type               TEXT NOT NULL DEFAULT 'train',
  backend_id         TEXT NOT NULL,
  status             TEXT NOT NULL,
  name               TEXT NOT NULL DEFAULT '',
  config             TEXT NOT NULL,
  progress           TEXT NOT NULL DEFAULT '{}',
  error              TEXT,
  cancel_requested   INTEGER NOT NULL DEFAULT 0,
  created_at         TEXT NOT NULL,
  started_at         TEXT,
  finished_at        TEXT
);
CREATE INDEX idx_jobs_project ON jobs(project_id);
CREATE INDEX idx_jobs_status  ON jobs(status);
CREATE TABLE job_events (
  job_id  TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  seq     INTEGER NOT NULL,
  ts      TEXT NOT NULL,
  type    TEXT NOT NULL,
  payload TEXT NOT NULL,
  PRIMARY KEY (job_id, seq)
);
CREATE TABLE checkpoints (
  id         TEXT PRIMARY KEY,
  job_id     TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  epoch      INTEGER NOT NULL,
  path       TEXT NOT NULL,
  metrics    TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);
CREATE INDEX idx_checkpoints_job ON checkpoints(job_id);
CREATE TABLE experiments (
  id                 TEXT PRIMARY KEY,
  project_id         TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  job_id             TEXT REFERENCES jobs(id) ON DELETE SET NULL,
  name               TEXT NOT NULL,
  task_type          TEXT NOT NULL,
  backend_id         TEXT NOT NULL,
  backend_version    TEXT NOT NULL DEFAULT '',
  architecture       TEXT NOT NULL DEFAULT '',
  seed               INTEGER,
  config             TEXT NOT NULL,
  dataset_id         TEXT,
  dataset_version_id TEXT,
  metrics            TEXT NOT NULL DEFAULT '{}',
  curves             TEXT NOT NULL DEFAULT '[]',
  artifacts          TEXT NOT NULL DEFAULT '[]',
  runtime            TEXT NOT NULL DEFAULT '{}',
  notes              TEXT NOT NULL DEFAULT '',
  created_at         TEXT NOT NULL
);
CREATE INDEX idx_experiments_project ON experiments(project_id);
CREATE TABLE models (
  id           TEXT PRIMARY KEY,
  project_id   TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  name         TEXT NOT NULL,
  task_type    TEXT NOT NULL,
  architecture TEXT NOT NULL DEFAULT '',
  notes        TEXT NOT NULL DEFAULT '',
  created_at   TEXT NOT NULL,
  updated_at   TEXT NOT NULL
);
CREATE INDEX idx_models_project ON models(project_id);
CREATE TABLE model_versions (
  id             TEXT PRIMARY KEY,
  model_id       TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
  version        INTEGER NOT NULL,
  experiment_id  TEXT,
  format         TEXT NOT NULL,
  file_path      TEXT NOT NULL,
  file_size      INTEGER NOT NULL,
  sha256         TEXT NOT NULL DEFAULT '',
  input_spec     TEXT NOT NULL DEFAULT '{}',
  output_spec    TEXT NOT NULL DEFAULT '{}',
  classes        TEXT NOT NULL DEFAULT '[]',
  metrics        TEXT NOT NULL DEFAULT '{}',
  export_formats TEXT NOT NULL DEFAULT '[]',
  license        TEXT NOT NULL DEFAULT '',
  runtime_reqs   TEXT NOT NULL DEFAULT '{}',
  notes          TEXT NOT NULL DEFAULT '',
  created_at     TEXT NOT NULL,
  UNIQUE(model_id, version)
);
CREATE INDEX idx_model_versions_model ON model_versions(model_id);
CREATE TABLE backend_installs (
  backend_id   TEXT PRIMARY KEY,
  status       TEXT NOT NULL,
  version      TEXT NOT NULL DEFAULT '',
  capabilities TEXT NOT NULL DEFAULT '{}',
  detail       TEXT NOT NULL DEFAULT '{}',
  checked_at   TEXT NOT NULL
);
`},
}

func (s *Store) migrate() error {
	if _, err := s.db.Exec(`CREATE TABLE IF NOT EXISTS schema_migrations (
	  id INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)`); err != nil {
		return err
	}
	var current int
	if err := s.db.QueryRow(`SELECT COALESCE(MAX(id),0) FROM schema_migrations`).Scan(&current); err != nil {
		return err
	}
	for _, m := range migrations {
		if m.id <= current {
			continue
		}
		tx, err := s.db.Begin()
		if err != nil {
			return err
		}
		if _, err := tx.Exec(m.sql); err != nil {
			tx.Rollback()
			return fmt.Errorf("migration %d: %w", m.id, err)
		}
		if _, err := tx.Exec(`INSERT INTO schema_migrations (id, applied_at) VALUES (?, ?)`,
			m.id, time.Now().UTC().Format(time.RFC3339)); err != nil {
			tx.Rollback()
			return err
		}
		if err := tx.Commit(); err != nil {
			return err
		}
	}
	return nil
}

// IntegrityCheck runs PRAGMA integrity_check and returns its result.
func (s *Store) IntegrityCheck(ctx context.Context) (string, error) {
	var res string
	err := s.db.QueryRowContext(ctx, `PRAGMA integrity_check`).Scan(&res)
	return res, err
}

// Backup copies the SQLite file to dest using the official backup approach:
// VACUUM INTO produces a compact, consistent snapshot even with WAL active.
func (s *Store) Backup(ctx context.Context, dest string) error {
	if _, err := os.Stat(dest); err == nil {
		return errors.New("backup destination already exists")
	}
	_, err := s.db.ExecContext(ctx, `VACUUM INTO ?`, dest)
	return err
}

// Restore swaps the live database file with src. The caller must ensure the
// process stops using the store first (NeuroStudio refuses to restore while
// jobs are running).
func Restore(livePath, src string) error {
	if _, err := os.Stat(src); err != nil {
		return err
	}
	// Keep a safety copy of the live db.
	stamp := time.Now().UTC().Format("20060102-150405")
	if err := os.Rename(livePath, livePath+".pre-restore-"+stamp); err != nil {
		return err
	}
	data, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	return os.WriteFile(livePath, data, 0o644)
}
