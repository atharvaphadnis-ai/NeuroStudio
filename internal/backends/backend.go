// Package backends defines the training-backend contract. A backend wraps a
// real training implementation (the built-in Go engine, or an external
// process such as a PyTorch worker). Backends declare their capabilities
// honestly: NeuroStudio never advertises a task a backend cannot run.
package backends

import (
	"context"
	"time"

	"neurostudio/internal/domain"
)

// TrainSpec carries everything a backend needs to run one training job.
type TrainSpec struct {
	JobID        string
	ProjectID    string
	ManifestPath string // dataset version manifest (classes, splits, paths)
	DataRoot     string // filesystem root the manifest paths are relative to
	WorkDir      string // job directory for checkpoints, logs, artifacts
	Config       domain.TrainConfig
}

// TrainResult describes produced artifacts.
type TrainResult struct {
	ModelPath    string
	Format       string // native-1 | onnx | ...
	Architecture string
	Classes      []string
	Input        domain.InputSpec
	Metrics      map[string]any
	Curves       []domain.CurvePt
	LogPath      string
}

// Compatibility explains whether a backend can run a given task.
type Compatibility struct {
	Supported bool   `json:"supported"`
	Reason    string `json:"reason,omitempty"`
}

// Report is the runtime discovery result for a backend.
type Report struct {
	ID           string                   `json:"id"`
	Name         string                   `json:"name"`
	Status       string                   `json:"status"` // available | unavailable
	Version      string                   `json:"version,omitempty"`
	InstallHint  string                   `json:"installHint,omitempty"`
	Tasks        map[string]Compatibility `json:"tasks"`
	Accelerators []string                 `json:"accelerators"`
	Detail       map[string]string        `json:"detail,omitempty"`
	CheckedAt    time.Time                `json:"checkedAt"`
}

// Backend is the one interface every training engine implements.
type Backend interface {
	ID() string
	Name() string
	Discover(ctx context.Context) Report
	Train(ctx context.Context, spec TrainSpec, emit func(domain.JobEvent)) (*TrainResult, error)
}

// Registry holds all known backends; availability is determined by Discover,
// never assumed.
type Registry struct {
	backends []Backend
}

func NewRegistry(bs ...Backend) *Registry { return &Registry{backends: bs} }

func (r *Registry) All() []Backend { return r.backends }

func (r *Registry) Get(id string) Backend {
	for _, b := range r.backends {
		if b.ID() == id {
			return b
		}
	}
	return nil
}

// DiscoverAll probes every backend (bounded time).
func (r *Registry) DiscoverAll(ctx context.Context) []Report {
	out := make([]Report, 0, len(r.backends))
	for _, b := range r.backends {
		out = append(out, b.Discover(ctx))
	}
	return out
}
