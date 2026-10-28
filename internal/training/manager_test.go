package training

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"encoding/json"
	"neurostudio/internal/app"
	"neurostudio/internal/backends"
	"neurostudio/internal/backends/goruntime"
	"neurostudio/internal/datasets"
	"neurostudio/internal/domain"
	"neurostudio/internal/projects"
	"neurostudio/internal/store"
)

// newTestStack wires a complete in-memory-on-disk environment: store,
// services, paths and the job manager with the native backend registered.
func newTestStack(t *testing.T) (*Manager, *projects.Service, *datasets.Service, app.Paths) {
	t.Helper()
	root := t.TempDir()
	st, err := store.Open(filepath.Join(root, "neurostudio.db"))
	if err != nil {
		t.Fatalf("open store: %v", err)
	}
	t.Cleanup(func() { st.Close() })

	paths := app.Paths{
		Root: root, Database: filepath.Join(root, "neurostudio.db"),
		Datasets: filepath.Join(root, "datasets"),
		Models:   filepath.Join(root, "models"),
		Jobs:     filepath.Join(root, "jobs"),
		Exports:  filepath.Join(root, "exports"),
		Backups:  filepath.Join(root, "backups"),
	}
	for _, d := range []string{paths.Datasets, paths.Models, paths.Jobs, paths.Exports, paths.Backups} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	reg := backends.NewRegistry(goruntime.New(nil))
	mgr := NewManager(st, reg, paths, nil)
	mgr.Start()
	t.Cleanup(mgr.Close)
	return mgr, projects.NewService(st), datasets.NewService(st, paths.Datasets), paths
}

// TestEndToEndTraining exercises the complete real workflow: synthetic
// dataset -> version/split -> enqueue -> real training -> experiment +
// registered model + evaluation metrics. No step is mocked.
func TestEndToEndTraining(t *testing.T) {
	mgr, pj, ds, paths := newTestStack(t)
	ctx := context.Background()

	proj, err := pj.Create(ctx, projects.CreateInput{Name: "E2E Shapes", TaskType: domain.TaskImageClassification})
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	d, res, err := ds.GenerateSampleDataset(ctx, proj.ID, "shapes", 2, 20, 7)
	if err != nil {
		t.Fatalf("generate dataset: %v", err)
	}
	if res.Added != 40 {
		t.Fatalf("expected 40 samples, got %d", res.Added)
	}
	dv, err := ds.CreateVersion(ctx, d.ID, domain.SplitParams{Train: 0.6, Val: 0.2, Test: 0.2, Seed: 7})
	if err != nil {
		t.Fatalf("create version: %v", err)
	}
	if dv.Stats.Counts["train"] == 0 || dv.Stats.Counts["test"] == 0 {
		t.Fatalf("split produced empty partitions: %+v", dv.Stats.Counts)
	}

	job, err := mgr.Enqueue(ctx, proj.ID, CreateInput{
		Name: "e2e-run", DatasetID: d.ID, VersionID: dv.ID, BackendID: "go-native",
		Config: domain.TrainConfig{
			TaskType: domain.TaskImageClassification, Architecture: "ns-cnn-small",
			Input:  domain.InputSpec{Size: 24, Grayscale: false},
			Epochs: 40, BatchSize: 8, LearningRate: 0.003, Optimizer: "adam",
			Seed: 42, Workers: 2,
			CheckpointEveryEpochs: 2,
			Augmentation:          domain.AugmentConfig{HFlip: true},
		},
	})
	if err != nil {
		t.Fatalf("enqueue: %v", err)
	}

	// Poll to completion (real training; bounded wait).
	deadline := time.Now().Add(120 * time.Second)
	for {
		j, err := mgr.GetJob(ctx, job.ID)
		if err != nil {
			t.Fatalf("get job: %v", err)
		}
		if j.Status == domain.JobCompleted || j.Status == domain.JobFailed || j.Status == domain.JobCancelled {
			if j.Status != domain.JobCompleted {
				t.Fatalf("job ended with status %s, error=%s", j.Status, j.Error)
			}
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("training did not finish in time; status=%s progress=%+v", j.Status, j.Progress)
		}
		time.Sleep(200 * time.Millisecond)
	}

	// Evaluation exists and is honest (computed on the held-out test split).
	events, err := mgr.Events(job.ID, 0)
	if err != nil {
		t.Fatalf("events: %v", err)
	}
	sawMetrics := false
	for _, e := range events {
		if e.Type == domain.EvtMetrics {
			sawMetrics = true
		}
	}
	if !sawMetrics {
		t.Fatal("no persisted epoch metrics events")
	}

	// Model artifact exists on disk.
	modelPath := filepath.Join(paths.Jobs, job.ID, "model.native.json")
	if _, err := os.Stat(modelPath); err != nil {
		t.Fatalf("model artifact missing: %v", err)
	}

	// The evaluation endpoint data: query the store like the API does.
	var metrics string
	if err := mgr.st.DB().QueryRow(`SELECT metrics FROM experiments WHERE job_id=?`, job.ID).Scan(&metrics); err != nil {
		t.Fatalf("experiment missing: %v", err)
	}
	if want := `"evalSplit":"test"`; !contains(metrics, want) {
		t.Fatalf("metrics should report evalSplit=test, got: %s", metrics)
	}
	// Shapes are highly separable; a real engine must learn them.
	acc := extractNumber(metrics, "evalAccuracy")
	if acc < 0.7 {
		t.Fatalf("eval accuracy %.3f too low for a separable synthetic dataset — engine may be broken", acc)
	}

	// Model registry entry exists with a downloadable file.
	var modelID, filePath string
	if err := mgr.st.DB().QueryRow(`SELECT id FROM models WHERE project_id=?`, proj.ID).Scan(&modelID); err != nil {
		t.Fatalf("model not registered: %v", err)
	}
	if err := mgr.st.DB().QueryRow(`SELECT file_path FROM model_versions WHERE model_id=?`, modelID).Scan(&filePath); err != nil {
		t.Fatalf("model version missing: %v", err)
	}
	if filePath != modelPath {
		t.Fatalf("registry points to %s, expected %s", filePath, modelPath)
	}
}

// TestEnqueueValidation covers the honest-refusal paths.
func TestEnqueueValidation(t *testing.T) {
	mgr, pj, ds, _ := newTestStack(t)
	ctx := context.Background()

	proj, _ := pj.Create(ctx, projects.CreateInput{Name: "V", TaskType: domain.TaskImageClassification})
	d, _, _ := ds.GenerateSampleDataset(ctx, proj.ID, "shapes", 2, 12, 3)
	dv, _ := ds.CreateVersion(ctx, d.ID, domain.SplitParams{Train: 0.6, Val: 0.2, Test: 0.2, Seed: 3})

	if _, err := mgr.Enqueue(ctx, proj.ID, CreateInput{DatasetID: d.ID, VersionID: dv.ID, BackendID: "pytorch-local"}); err == nil {
		t.Fatal("unknown backend must be rejected")
	}
	if _, err := mgr.Enqueue(ctx, proj.ID, CreateInput{DatasetID: "missing", VersionID: dv.ID, BackendID: "go-native"}); err == nil {
		t.Fatal("mismatched dataset/version must be rejected")
	}
	if _, err := mgr.Enqueue(ctx, "missing-project", CreateInput{DatasetID: d.ID, VersionID: dv.ID, BackendID: "go-native"}); err == nil {
		t.Fatal("unknown project must be rejected")
	}
}

func contains(hay, needle string) bool {
	return len(hay) >= len(needle) && (hay == needle || indexOf(hay, needle) >= 0)
}

func indexOf(h, n string) int {
	for i := 0; i+len(n) <= len(h); i++ {
		if h[i:i+len(n)] == n {
			return i
		}
	}
	return -1
}

func extractNumber(metricsJSON, key string) float64 {
	var m map[string]any
	_ = jsonUnmarshal(metricsJSON, &m)
	if v, ok := m[key].(float64); ok {
		return v
	}
	return -1
}

func jsonUnmarshal(s string, v any) error {
	return json.Unmarshal([]byte(s), v)
}
