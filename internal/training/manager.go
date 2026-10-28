// Package training implements the job manager: a single-worker orchestrator
// that owns the lifecycle of training jobs. It persists jobs and their event
// streams in SQLite, runs one real training at a time (the native engine is
// CPU-bound and single-user by design), streams live updates to SSE
// clients, honors cancellation, and on success records an experiment and
// registers the produced model in the registry.
package training

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sync"
	"time"

	"neurostudio/internal/app"
	"neurostudio/internal/backends"
	"neurostudio/internal/domain"
	"neurostudio/internal/store"
	"neurostudio/internal/util"
)

// Manager orchestrates training jobs.
type Manager struct {
	st    *store.Store
	reg   *backends.Registry
	paths app.Paths
	log   *slog.Logger

	mu      sync.Mutex
	hub     *Hub
	seq     map[string]int64 // jobID -> last persisted event seq
	cancels map[string]context.CancelFunc
	queue   chan string
	wake    chan struct{}
	quit    chan struct{}
	started bool
}

func NewManager(st *store.Store, reg *backends.Registry, paths app.Paths, log *slog.Logger) *Manager {
	return &Manager{
		st: st, reg: reg, paths: paths, log: log,
		hub:     NewHub(),
		seq:     map[string]int64{},
		cancels: map[string]context.CancelFunc{},
		queue:   make(chan string, 256),
		wake:    make(chan struct{}, 1),
		quit:    make(chan struct{}),
	}
}

// Registry exposes the backend registry (capability display).
func (m *Manager) Registry() *backends.Registry { return m.reg }

// Hub exposes the live event hub (SSE fan-out).
func (m *Manager) Hub() *Hub { return m.hub }

// Start launches the single worker goroutine.
func (m *Manager) Start() {
	if m.started {
		return
	}
	m.started = true
	go m.worker()
}

// Close stops the worker (running jobs are allowed to finish; the process
// shutdown path marks them failed on next boot as documented).
func (m *Manager) Close() { close(m.quit) }

// ---- job creation ----

type CreateInput struct {
	Name      string             `json:"name"`
	DatasetID string             `json:"datasetId"`
	VersionID string             `json:"versionId"`
	BackendID string             `json:"backendId"`
	Config    domain.TrainConfig `json:"config"`
}

// Enqueue validates the request against reality (backend capabilities,
// dataset existence, manifest availability) and persists a queued job.
func (m *Manager) Enqueue(ctx context.Context, projectID string, in CreateInput) (*domain.Job, error) {
	// Project must exist.
	var exists int
	if err := m.st.DB().QueryRowContext(ctx, `SELECT 1 FROM projects WHERE id=?`, projectID).Scan(&exists); err != nil {
		if err == sql.ErrNoRows {
			return nil, util.ErrNotFound("project")
		}
		return nil, err
	}
	if in.DatasetID == "" || in.VersionID == "" {
		return nil, util.ErrBadRequest("datasetId and versionId are required")
	}
	if in.BackendID == "" {
		in.BackendID = "go-native"
	}
	backend := m.reg.Get(in.BackendID)
	if backend == nil {
		return nil, util.ErrBadRequest("unknown backend %q — see GET /api/v1/training/backends", in.BackendID)
	}

	// Dataset version + manifest must exist on disk.
	var datasetID, manifestPath string
	err := m.st.DB().QueryRowContext(ctx,
		`SELECT dv.dataset_id, dv.manifest_path FROM dataset_versions dv
                 JOIN datasets d ON d.id = dv.dataset_id
                 WHERE dv.id=? AND d.project_id=?`, in.VersionID, projectID).
		Scan(&datasetID, &manifestPath)
	if err == sql.ErrNoRows {
		return nil, util.ErrNotFound("dataset version (it must belong to this project)")
	}
	if err != nil {
		return nil, err
	}
	if datasetID != in.DatasetID {
		return nil, util.ErrBadRequest("versionId does not belong to datasetId")
	}
	if _, err := os.Stat(manifestPath); err != nil {
		return nil, util.Err(409, "manifest_missing", "dataset manifest file is missing; recreate the dataset version")
	}

	// Honest capability check: does this backend really support the task?
	report := backend.Discover(ctx)
	task := in.Config.TaskType
	if task == "" {
		task = domain.TaskImageClassification
	}
	compat, ok := report.Tasks[string(task)]
	if !ok || !compat.Supported {
		return nil, util.Err(422, "backend_unsupported",
			fmt.Sprintf("backend %q does not support task %q", in.BackendID, task))
	}
	if in.Config.Architecture == "" {
		in.Config.Architecture = "ns-cnn-small"
	}
	if in.Config.Architecture != "ns-cnn-small" {
		return nil, util.ErrBadRequest("unknown architecture %q (supported: ns-cnn-small)", in.Config.Architecture)
	}
	in.Config.TaskType = task
	if in.Config.Input.Size == 0 {
		in.Config.Input.Size = 48
	}

	name := in.Name
	if name == "" {
		name = "train-" + time.Now().UTC().Format("0102-150405")
	}

	job := &domain.Job{
		ID: util.NewID(), ProjectID: projectID,
		DatasetID: in.DatasetID, DatasetVersionID: in.VersionID,
		Type: "train", BackendID: in.BackendID, Status: domain.JobQueued,
		Name: name, Config: in.Config, CreatedAt: time.Now().UTC(),
	}
	cfgJSON, _ := json.Marshal(in.Config)
	defer m.st.Lock()()
	_, err = m.st.DB().ExecContext(ctx, `INSERT INTO jobs
          (id, project_id, dataset_id, dataset_version_id, type, backend_id, status, name, config, progress, created_at)
          VALUES (?,?,?,?,?,?,?,?,?, '{}', ?)`,
		job.ID, job.ProjectID, job.DatasetID, job.DatasetVersionID, job.Type,
		job.BackendID, string(job.Status), job.Name, string(cfgJSON),
		job.CreatedAt.Format(time.RFC3339))
	if err != nil {
		return nil, err
	}
	select {
	case m.queue <- job.ID:
	default:
		// Queue full: refuse rather than silently drop.
		m.st.DB().ExecContext(ctx, `UPDATE jobs SET status='failed', error='job queue full' WHERE id=?`, job.ID)
		return nil, util.Err(503, "queue_full", "too many queued jobs; try again shortly")
	}
	return job, nil
}

// ---- worker ----

func (m *Manager) worker() {
	for {
		select {
		case <-m.quit:
			return
		case id := <-m.queue:
			m.run(id)
		case <-m.wake:
			// reserved for future re-queue paths
		}
	}
}

func (m *Manager) run(jobID string) {
	ctx, cancel := context.WithCancel(context.Background())
	m.mu.Lock()
	m.cancels[jobID] = cancel
	m.mu.Unlock()
	defer func() {
		cancel()
		m.mu.Lock()
		delete(m.cancels, jobID)
		m.mu.Unlock()
	}()

	job, err := m.getJob(jobID)
	if err != nil {
		m.log.Error("job vanished", "job", jobID, "err", err)
		return
	}
	now := time.Now().UTC()
	m.persistJobState(jobID, domain.JobRunning, "", &now, nil)
	m.publish(domain.JobEvent{JobID: jobID, Type: domain.EvtStatus,
		Payload: map[string]string{"status": string(domain.JobRunning)}, Time: now})

	// Resolve manifest path + work dir.
	var manifestPath string
	if err := m.st.DB().QueryRow(`SELECT manifest_path FROM dataset_versions WHERE id=?`,
		job.DatasetVersionID).Scan(&manifestPath); err != nil {
		m.fail(jobID, "dataset version manifest unavailable: "+err.Error())
		return
	}
	workDir := filepath.Join(m.paths.Jobs, jobID)
	if err := os.MkdirAll(workDir, 0o755); err != nil {
		m.fail(jobID, "cannot create job directory: "+err.Error())
		return
	}

	backend := m.reg.Get(job.BackendID)
	if backend == nil {
		m.fail(jobID, "backend disappeared: "+job.BackendID)
		return
	}

	spec := backends.TrainSpec{
		JobID: jobID, ProjectID: job.ProjectID,
		ManifestPath: manifestPath, DataRoot: m.paths.Datasets, WorkDir: workDir,
		Config: job.Config,
	}

	startedAt := time.Now()
	res, trainErr := backend.Train(ctx, spec, func(evt domain.JobEvent) {
		evt.JobID = jobID
		evt.Time = time.Now().UTC()
		m.onEvent(jobID, evt, startedAt)
	})

	finished := time.Now().UTC()
	switch {
	case trainErr == nil:
		if err := m.complete(job, res, finished); err != nil {
			m.log.Error("post-training registration failed", "job", jobID, "err", err)
			m.fail(jobID, "training finished but registration failed: "+err.Error())
		}
	case trainErr == util.ErrCancelled:
		m.persistJobState(jobID, domain.JobCancelled, "cancelled by user", &finished, nil)
		m.publish(domain.JobEvent{JobID: jobID, Type: domain.EvtStatus,
			Payload: map[string]string{"status": string(domain.JobCancelled)}, Time: finished})
		m.publish(domain.JobEvent{JobID: jobID, Type: domain.EvtDone,
			Payload: map[string]string{"status": string(domain.JobCancelled)}, Time: finished})
	default:
		msg := trainErr.Error()
		m.log.Warn("training failed", "job", jobID, "err", msg)
		m.fail(jobID, msg)
	}
}

// onEvent persists selected events, updates stored progress and broadcasts.
func (m *Manager) onEvent(jobID string, evt domain.JobEvent, startedAt time.Time) {
	// Live-only tick events are broadcast but not persisted.
	if evt.Type == "tick" {
		m.hub.Publish(evt)
		return
	}

	m.mu.Lock()
	seq := m.seq[jobID] + 1
	m.seq[jobID] = seq
	m.mu.Unlock()
	evt.Seq = seq

	payload, _ := json.Marshal(evt.Payload)
	if evt.Type == domain.EvtMetrics || evt.Type == domain.EvtPhase {
		m.updateProgress(jobID, evt, startedAt)
	}
	if evt.Type == domain.EvtCheckpoint {
		if rec, ok := evt.Payload.(map[string]any); ok {
			var epoch int
			switch v := rec["epoch"].(type) {
			case float64:
				epoch = int(v)
			case int:
				epoch = v
			}
			path, _ := rec["path"].(string)
			metrics := map[string]float64{}
			if v, ok := rec["valLoss"].(float64); ok {
				metrics["valLoss"] = v
			}
			if v, ok := rec["valAcc"].(float64); ok {
				metrics["valAccuracy"] = v
			}
			mj, _ := json.Marshal(metrics)
			unlock := m.st.Lock()
			m.st.DB().Exec(`INSERT OR IGNORE INTO checkpoints (id, job_id, epoch, path, metrics, created_at)
                          VALUES (?,?,?,?,?,?)`, util.NewID(), jobID, int(epoch), path, string(mj),
				time.Now().UTC().Format(time.RFC3339))
			unlock()
		}
	}

	unlock := m.st.Lock()
	m.st.DB().Exec(`INSERT INTO job_events (job_id, seq, ts, type, payload) VALUES (?,?,?,?,?)`,
		jobID, seq, evt.Time.Format(time.RFC3339Nano), evt.Type, string(payload))
	unlock()

	m.hub.Publish(evt)
}

func (m *Manager) updateProgress(jobID string, evt domain.JobEvent, startedAt time.Time) {
	var p domain.Progress
	var cur string
	if err := m.st.DB().QueryRow(`SELECT progress FROM jobs WHERE id=?`, jobID).Scan(&cur); err == nil {
		_ = json.Unmarshal([]byte(cur), &p)
	}
	switch evt.Type {
	case domain.EvtPhase:
		if rec, ok := evt.Payload.(map[string]string); ok {
			p.Phase = rec["phase"]
		}
	case domain.EvtMetrics:
		if em, ok := evt.Payload.(domain.EpochMetrics); ok {
			p.Phase = "train"
			p.Epoch = em.Epoch
			p.Epochs = em.Epochs
			p.Step = em.Step
			p.StepsPerEpoch = em.Steps
			p.TrainLoss = em.TrainLoss
			p.ValLoss = em.ValLoss
			p.ValAccuracy = em.ValAcc
			p.LR = em.LR
			if em.ValLoss > 0 && (p.BestValLoss == 0 || em.ValLoss < p.BestValLoss) {
				p.BestValLoss = em.ValLoss
				p.BestEpoch = em.Epoch
			}
			p.Curves = append(p.Curves, domain.CurvePt{
				Epoch: em.Epoch, TrainLoss: em.TrainLoss, ValLoss: em.ValLoss,
				ValAcc: em.ValAcc, LR: em.LR,
			})
			if len(p.Curves) > 500 {
				p.Curves = p.Curves[len(p.Curves)-500:]
			}
			elapsed := time.Since(startedAt).Seconds()
			p.ElapsedMS = int64(elapsed * 1000)
			if em.Epoch > 0 {
				per := elapsed / float64(em.Epoch)
				p.ETASeconds = per * float64(em.Epochs-em.Epoch)
			}
		}
	}
	pj, _ := json.Marshal(p)
	m.st.DB().Exec(`UPDATE jobs SET progress=? WHERE id=?`, string(pj), jobID)
}

func (m *Manager) complete(job *domain.Job, res *backends.TrainResult, finished time.Time) error {
	progress := domain.Progress{
		Phase: "done", Epoch: len(res.Curves), Epochs: job.Config.Epochs,
		BestValLoss: num(res.Metrics["bestValLoss"]), BestEpoch: int(num(res.Metrics["bestEpoch"])),
		Curves: res.Curves,
	}
	progressJSON, _ := json.Marshal(progress)
	metricsJSON, _ := json.Marshal(res.Metrics)
	curvesJSON, _ := json.Marshal(res.Curves)

	unlock := m.st.Lock()
	defer unlock()
	_, err := m.st.DB().Exec(`UPDATE jobs SET status=?, finished_at=?, progress=? WHERE id=?`,
		string(domain.JobCompleted), finished.Format(time.RFC3339), string(progressJSON), job.ID)
	if err != nil {
		return err
	}

	// Experiment record — the audit trail of what was run and what it scored.
	expID := util.NewID()
	artifacts := []domain.ArtifactRef{}
	if st, err := os.Stat(res.ModelPath); err == nil {
		artifacts = append(artifacts, domain.ArtifactRef{Kind: "model", Path: res.ModelPath, Size: st.Size(), Name: filepath.Base(res.ModelPath)})
	}
	artJSON, _ := json.Marshal(artifacts)
	expName := job.Name
	_, err = m.st.DB().Exec(`INSERT INTO experiments
          (id, project_id, job_id, name, task_type, backend_id, backend_version, architecture, seed,
           config, dataset_id, dataset_version_id, metrics, curves, artifacts, runtime, created_at)
          VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`,
		expID, job.ProjectID, job.ID, expName, string(job.Config.TaskType), job.BackendID,
		"1.0.0", res.Architecture, job.Config.Seed, mustJSON(job.Config),
		job.DatasetID, job.DatasetVersionID, metricsJSON, curvesJSON, artJSON,
		mustJSON(map[string]any{"engine": "go-native", "os": "embedded"}), finished.Format(time.RFC3339))
	if err != nil {
		return fmt.Errorf("insert experiment: %w", err)
	}

	// Model registry entry: model + version 1 (a new model per completed job;
	// users can rename and organize from the UI).
	st, err := os.Stat(res.ModelPath)
	if err != nil {
		return fmt.Errorf("model artifact missing: %w", err)
	}
	sum, _ := fileSHA256(res.ModelPath)
	modelID := util.NewID()
	_, err = m.st.DB().Exec(`INSERT INTO models (id, project_id, name, task_type, architecture, notes, created_at, updated_at)
          VALUES (?,?,?,?,?,?,?,?)`,
		modelID, job.ProjectID, expName, string(job.Config.TaskType), res.Architecture,
		"Registered automatically from job "+job.ID, finished.Format(time.RFC3339), finished.Format(time.RFC3339))
	if err != nil {
		return fmt.Errorf("insert model: %w", err)
	}
	inputJSON, _ := json.Marshal(res.Input)
	exportFormats := []string{"native-1"}
	efJSON, _ := json.Marshal(exportFormats)
	classesJSON, _ := json.Marshal(res.Classes)
	_, err = m.st.DB().Exec(`INSERT INTO model_versions
          (id, model_id, version, experiment_id, format, file_path, file_size, sha256,
           input_spec, output_spec, classes, metrics, export_formats, license, runtime_reqs, notes, created_at)
          VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`,
		util.NewID(), modelID, 1, expID, "native-1", res.ModelPath, st.Size(), sum,
		inputJSON, mustJSON(map[string]any{"type": "class_probabilities", "classes": res.Classes}),
		classesJSON, metricsJSON, efJSON,
		"Provided by the model creator — user dataset, user responsibility",
		mustJSON(map[string]any{"runtime": "neurostudio go-native", "minVersion": "0.1.0"}),
		"Trained by job "+job.ID, finished.Format(time.RFC3339))
	if err != nil {
		return fmt.Errorf("insert model version: %w", err)
	}

	m.publish(domain.JobEvent{JobID: job.ID, Type: domain.EvtStatus,
		Payload: map[string]string{"status": string(domain.JobCompleted), "experimentId": expID, "modelId": modelID}, Time: finished})
	m.publish(domain.JobEvent{JobID: job.ID, Type: domain.EvtDone,
		Payload: map[string]any{"status": "completed", "experimentId": expID, "modelId": modelID}, Time: finished})
	return nil
}

func (m *Manager) fail(jobID, msg string) {
	finished := time.Now().UTC()
	m.persistJobState(jobID, domain.JobFailed, msg, &finished, nil)
	m.publish(domain.JobEvent{JobID: jobID, Type: domain.EvtError,
		Payload: map[string]string{"error": msg}, Time: finished})
	m.publish(domain.JobEvent{JobID: jobID, Type: domain.EvtDone,
		Payload: map[string]string{"status": string(domain.JobFailed), "error": msg}, Time: finished})
}

func (m *Manager) persistJobState(jobID string, status domain.JobStatus, errMsg string, finishedAt *time.Time, startedAt *time.Time) {
	defer m.st.Lock()()
	if startedAt != nil {
		m.st.DB().Exec(`UPDATE jobs SET status=?, started_at=? WHERE id=?`,
			string(status), startedAt.Format(time.RFC3339), jobID)
		return
	}
	if finishedAt != nil {
		m.st.DB().Exec(`UPDATE jobs SET status=?, error=?, finished_at=? WHERE id=?`,
			string(status), errMsg, finishedAt.Format(time.RFC3339), jobID)
		return
	}
	m.st.DB().Exec(`UPDATE jobs SET status=? WHERE id=?`, string(status), jobID)
}

// ---- public queries / controls ----

func (m *Manager) getJob(id string) (*domain.Job, error) {
	row := m.st.DB().QueryRow(`SELECT id, project_id, dataset_id, dataset_version_id, type, backend_id,
          status, name, config, progress, COALESCE(error,''), cancel_requested, created_at, started_at, finished_at
          FROM jobs WHERE id=?`, id)
	return scanJob(row)
}

func scanJob(sc interface{ Scan(...any) error }) (*domain.Job, error) {
	var j domain.Job
	var cfg, progress string
	var status string
	var created string
	var started, finished sql.NullString
	var cancelReq int
	if err := sc.Scan(&j.ID, &j.ProjectID, &j.DatasetID, &j.DatasetVersionID, &j.Type, &j.BackendID,
		&status, &j.Name, &cfg, &progress, &j.Error, &cancelReq, &created, &started, &finished); err != nil {
		if err == sql.ErrNoRows {
			return nil, util.ErrNotFound("job")
		}
		return nil, err
	}
	j.Status = domain.JobStatus(status)
	j.CancelRequested = cancelReq == 1
	_ = json.Unmarshal([]byte(cfg), &j.Config)
	_ = json.Unmarshal([]byte(progress), &j.Progress)
	j.CreatedAt, _ = time.Parse(time.RFC3339, created)
	if started.Valid {
		t, _ := time.Parse(time.RFC3339, started.String)
		j.StartedAt = &t
	}
	if finished.Valid {
		t, _ := time.Parse(time.RFC3339, finished.String)
		j.FinishedAt = &t
	}
	return &j, nil
}

// GetJob returns one job (NotFound if unknown).
func (m *Manager) GetJob(ctx context.Context, id string) (*domain.Job, error) {
	return m.getJob(id)
}

// ListJobs returns the project's jobs, newest first.
func (m *Manager) ListJobs(ctx context.Context, projectID string, limit int) ([]*domain.Job, error) {
	if limit <= 0 || limit > 200 {
		limit = 100
	}
	rows, err := m.st.DB().QueryContext(ctx, `SELECT id, project_id, dataset_id, dataset_version_id, type, backend_id,
          status, name, config, progress, COALESCE(error,''), cancel_requested, created_at, started_at, finished_at
          FROM jobs WHERE project_id=? ORDER BY created_at DESC LIMIT ?`, projectID, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []*domain.Job{}
	for rows.Next() {
		j, err := scanJob(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, j)
	}
	return out, rows.Err()
}

// Cancel requests cooperative cancellation. Queued jobs are cancelled
// immediately; running jobs stop at the next batch boundary.
func (m *Manager) Cancel(ctx context.Context, id string) (*domain.Job, error) {
	job, err := m.getJob(id)
	if err != nil {
		return nil, err
	}
	switch job.Status {
	case domain.JobCompleted, domain.JobFailed, domain.JobCancelled:
		return job, nil // already finished; idempotent
	case domain.JobQueued:
		finished := time.Now().UTC()
		m.persistJobState(id, domain.JobCancelled, "cancelled while queued", &finished, nil)
		return m.getJob(id)
	default: // running
		unlock := m.st.Lock()
		m.st.DB().Exec(`UPDATE jobs SET cancel_requested=1 WHERE id=? AND status='running'`, id)
		unlock()
		m.mu.Lock()
		cancel := m.cancels[id]
		m.mu.Unlock()
		if cancel != nil {
			m.publish(domain.JobEvent{JobID: id, Type: domain.EvtLog, Time: time.Now().UTC(),
				Payload: map[string]string{"message": "cancellation requested — stopping at the next batch boundary"}})
			cancel()
		}
		return m.getJob(id)
	}
}

// Events replays persisted events after the given seq.
func (m *Manager) Events(jobID string, afterSeq int64) ([]domain.JobEvent, error) {
	rows, err := m.st.DB().Query(`SELECT seq, ts, type, payload FROM job_events
          WHERE job_id=? AND seq>? ORDER BY seq`, jobID, afterSeq)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []domain.JobEvent{}
	for rows.Next() {
		var evt domain.JobEvent
		var ts, payload string
		if err := rows.Scan(&evt.Seq, &ts, &evt.Type, &payload); err != nil {
			return nil, err
		}
		evt.JobID = jobID
		evt.Time, _ = time.Parse(time.RFC3339Nano, ts)
		_ = json.Unmarshal([]byte(payload), &evt.Payload)
		out = append(out, evt)
	}
	return out, rows.Err()
}

// LatestSeq returns the highest persisted event seq for a job (0 if none).
func (m *Manager) LatestSeq(jobID string) int64 {
	var seq sql.NullInt64
	m.st.DB().QueryRow(`SELECT MAX(seq) FROM job_events WHERE job_id=?`, jobID).Scan(&seq)
	return seq.Int64
}

// ---- SSE hub ----

// Hub fans live events out to any number of SSE subscribers per job.
type Hub struct {
	mu   sync.Mutex
	subs map[string]map[chan domain.JobEvent]struct{}
}

func NewHub() *Hub { return &Hub{subs: map[string]map[chan domain.JobEvent]struct{}{}} }

func (h *Hub) Subscribe(jobID string) (<-chan domain.JobEvent, func()) {
	ch := make(chan domain.JobEvent, 256)
	h.mu.Lock()
	if h.subs[jobID] == nil {
		h.subs[jobID] = map[chan domain.JobEvent]struct{}{}
	}
	h.subs[jobID][ch] = struct{}{}
	h.mu.Unlock()
	off := func() {
		h.mu.Lock()
		delete(h.subs[jobID], ch)
		if len(h.subs[jobID]) == 0 {
			delete(h.subs, jobID)
		}
		h.mu.Unlock()
	}
	return ch, off
}

func (h *Hub) Publish(evt domain.JobEvent) {
	h.mu.Lock()
	defer h.mu.Unlock()
	for ch := range h.subs[evt.JobID] {
		select {
		case ch <- evt:
		default: // slow consumer: drop rather than block training
		}
	}
}

// ---- helpers ----

func (m *Manager) publish(evt domain.JobEvent) { m.hub.Publish(evt) }

func num(v any) float64 {
	switch x := v.(type) {
	case float64:
		return x
	case int:
		return float64(x)
	}
	return 0
}

func mustJSON(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		return "{}"
	}
	return string(b)
}

func fileSHA256(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), nil
}
