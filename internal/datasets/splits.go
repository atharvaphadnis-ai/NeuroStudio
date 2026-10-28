package datasets

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"time"

	"neurostudio/internal/domain"
	"neurostudio/internal/util"
)

func clockNowUTC() time.Time { return time.Now().UTC() }

// CreateVersion builds a new dataset version: a reproducible, seeded,
// stratified train/val/test split plus a manifest file. Only 'ok' samples
// participate; duplicates and corrupt samples are excluded from splits and
// reported in stats.
func (s *Service) CreateVersion(ctx context.Context, datasetID string, p domain.SplitParams) (*domain.DatasetVersion, error) {
	ds, err := s.Get(ctx, datasetID)
	if err != nil {
		return nil, err
	}
	if p.Train <= 0 || p.Val < 0 || p.Test < 0 || p.Train+p.Val+p.Test > 1.0001 || p.Train+p.Val+p.Test < 0.9999 {
		return nil, util.ErrBadRequest("split fractions must sum to 1.0 (e.g. 0.7/0.15/0.15)")
	}
	if ds.DataType != domain.DataImage {
		return nil, util.ErrBadRequest("splits are currently supported for image datasets")
	}
	classes, err := s.ListClasses(ctx, datasetID)
	if err != nil {
		return nil, err
	}
	if len(classes) < 2 {
		return nil, util.ErrBadRequest("dataset needs at least 2 classes before splitting")
	}

	// Collect usable samples per class (status = ok, class assigned).
	perClass := map[string][]domain.Sample{} // classID -> samples
	var totalBytes int64
	rows, err := s.st.DB().QueryContext(ctx, `SELECT s.id, s.dataset_id, s.class_id, COALESCE(c.name,''), s.filename, s.rel_path,
          s.media_type, s.size_bytes, s.sha256, s.width, s.height, s.duration_ms, s.status, s.meta, s.created_at
          FROM samples s LEFT JOIN classes c ON s.class_id=c.id WHERE s.dataset_id=? AND s.status='ok'`, datasetID)
	if err != nil {
		return nil, err
	}
	n := 0
	for rows.Next() {
		sm, err := scanSample(rows)
		if err != nil {
			rows.Close()
			return nil, err
		}
		if sm.ClassID == nil {
			continue
		}
		perClass[*sm.ClassID] = append(perClass[*sm.ClassID], *sm)
		totalBytes += sm.SizeBytes
		n++
	}
	rows.Close()
	if n < 4 {
		return nil, util.ErrBadRequest("dataset needs at least 4 labeled valid samples (has %d)", n)
	}

	// Seeded stratified assignment.
	rng := rand.New(rand.NewSource(p.Seed))
	splits := map[string]string{}
	for _, c := range classes {
		samples := perClass[c.ID]
		if len(samples) == 0 {
			continue
		}
		rng.Shuffle(len(samples), func(i, j int) { samples[i], samples[j] = samples[j], samples[i] })
		nTrain := int(float64(len(samples)) * p.Train)
		nVal := int(float64(len(samples)) * p.Val)
		if nTrain < 1 {
			nTrain = 1
		}
		if nVal < 1 && p.Val > 0 && len(samples) > 2 {
			nVal = 1
		}
		for i, sm := range samples {
			switch {
			case i < nTrain:
				splits[sm.ID] = domain.SplitTrain
			case i < nTrain+nVal:
				splits[sm.ID] = domain.SplitVal
			default:
				splits[sm.ID] = domain.SplitTest
			}
		}
	}

	stats := computeVersionStats(splits, perClass, classes)

	// Leakage detection: identical content hashes appearing in more than one
	// split would leak information between train and eval.
	hashSplit := map[string]string{}
	leaks := 0
	for _, c := range classes {
		for _, sm := range perClass[c.ID] {
			sp, ok := splits[sm.ID]
			if !ok {
				continue
			}
			if first, seen := hashSplit[sm.SHA256]; seen && first != sp {
				leaks++
			} else if !seen {
				hashSplit[sm.SHA256] = sp
			}
		}
	}
	if leaks > 0 {
		stats.Warnings = append(stats.Warnings, fmt.Sprintf(
			"%d samples share file content across different splits (possible data leakage — remove duplicates)", leaks))
	}
	if len(stats.ByClass) < 2 {
		stats.Warnings = append(stats.Warnings, "fewer than 2 classes contain samples — training requires at least 2")
	}

	// Version number.
	var version int
	defer s.st.Lock()()
	if err := s.st.DB().QueryRowContext(ctx,
		`SELECT COALESCE(MAX(version),0)+1 FROM dataset_versions WHERE dataset_id=?`, datasetID).Scan(&version); err != nil {
		return nil, err
	}

	manifest := s.buildManifest(ds, &version, classes, perClass, splits, p)
	manifestDir := filepath.Join(s.dataRoot, datasetID, "versions")
	if err := os.MkdirAll(manifestDir, 0o755); err != nil {
		return nil, err
	}
	manifestPath := filepath.Join(manifestDir, fmt.Sprintf("v%d.json", version))
	manifestBytes, _ := json.MarshalIndent(manifest, "", " ")
	if err := util.AtomicWrite(manifestPath, manifestBytes, 0o644); err != nil {
		return nil, err
	}

	splitsJSON, _ := json.Marshal(splits)
	paramsJSON, _ := json.Marshal(p)
	statsJSON, _ := json.Marshal(stats)
	dv := &domain.DatasetVersion{
		ID: util.NewID(), DatasetID: datasetID, Version: version, Seed: p.Seed,
		Params: p, Stats: *stats, ManifestPath: manifestPath, Splits: splits,
		CreatedAt: clockNowUTC(),
	}
	_, err = s.st.DB().ExecContext(ctx, `INSERT INTO dataset_versions
          (id, dataset_id, version, seed, params, splits, manifest_path, stats, created_at)
          VALUES (?,?,?,?,?,?,?,?,?)`,
		dv.ID, dv.DatasetID, dv.Version, dv.Seed, string(paramsJSON), string(splitsJSON),
		manifestPath, string(statsJSON), dv.CreatedAt.Format("2006-01-02T15:04:05Z"))
	if err != nil {
		return nil, err
	}
	return dv, nil
}

type Manifest struct {
	Format      string             `json:"format"` // "neurostudio/manifest-1"
	DatasetID   string             `json:"datasetId"`
	DatasetName string             `json:"datasetName"`
	DataType    string             `json:"dataType"`
	Version     int                `json:"version"`
	Seed        int64              `json:"seed"`
	Params      domain.SplitParams `json:"params"`
	Classes     []string           `json:"classes"`
	Samples     []ManifestSample   `json:"samples"`
	CreatedAt   string             `json:"createdAt"`
}

type ManifestSample struct {
	ID    string `json:"id"`
	Path  string `json:"path"` // dataset-relative
	Class string `json:"class"`
	Split string `json:"split"`
	SHA   string `json:"sha256"`
	W     int    `json:"w,omitempty"`
	H     int    `json:"h,omitempty"`
}

func (s *Service) buildManifest(ds *domain.Dataset, version *int, classes []domain.Class,
	perClass map[string][]domain.Sample, splits map[string]string, p domain.SplitParams) *Manifest {
	m := &Manifest{
		Format: "neurostudio/manifest-1", DatasetID: ds.ID, DatasetName: ds.Name,
		DataType: string(ds.DataType), Version: *version, Seed: p.Seed, Params: p,
		CreatedAt: clockNowUTC().Format("2006-01-02T15:04:05Z"),
	}
	for _, c := range classes {
		m.Classes = append(m.Classes, c.Name)
	}
	for _, c := range classes {
		for _, sm := range perClass[c.ID] {
			sp, ok := splits[sm.ID]
			if !ok {
				continue
			}
			m.Samples = append(m.Samples, ManifestSample{
				ID: sm.ID, Path: sm.RelPath, Class: c.Name, Split: sp,
				SHA: sm.SHA256, W: sm.Width, H: sm.Height,
			})
		}
	}
	sort.Slice(m.Samples, func(i, j int) bool { return m.Samples[i].ID < m.Samples[j].ID })
	return m
}

func computeVersionStats(splits map[string]string, perClass map[string][]domain.Sample, classes []domain.Class) *domain.DatasetVersionStats {
	st := &domain.DatasetVersionStats{
		Counts:  map[string]int{"train": 0, "val": 0, "test": 0},
		ByClass: map[string]map[string]int{},
	}
	for _, c := range classes {
		st.ByClass[c.Name] = map[string]int{"train": 0, "val": 0, "test": 0}
		for _, sm := range perClass[c.ID] {
			if sp, ok := splits[sm.ID]; ok {
				st.Counts[sp]++
				st.ByClass[c.Name][sp]++
			}
		}
	}
	return st
}

func (s *Service) ListVersions(ctx context.Context, datasetID string) ([]*domain.DatasetVersion, error) {
	rows, err := s.st.DB().QueryContext(ctx, `SELECT id, dataset_id, version, seed, params, splits, manifest_path, stats, created_at
          FROM dataset_versions WHERE dataset_id=? ORDER BY version DESC`, datasetID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []*domain.DatasetVersion{}
	for rows.Next() {
		dv, err := scanVersion(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, dv)
	}
	return out, rows.Err()
}

func (s *Service) GetVersion(ctx context.Context, id string) (*domain.DatasetVersion, error) {
	row := s.st.DB().QueryRowContext(ctx, `SELECT id, dataset_id, version, seed, params, splits, manifest_path, stats, created_at
          FROM dataset_versions WHERE id=?`, id)
	dv, err := scanVersion(row)
	if err == sql.ErrNoRows {
		return nil, util.ErrNotFound("dataset version")
	}
	return dv, err
}

func scanVersion(sc interface{ Scan(...any) error }) (*domain.DatasetVersion, error) {
	var dv domain.DatasetVersion
	var params, splits, stats string
	var created string
	if err := sc.Scan(&dv.ID, &dv.DatasetID, &dv.Version, &dv.Seed, &params, &splits,
		&dv.ManifestPath, &stats, &created); err != nil {
		return nil, err
	}
	_ = json.Unmarshal([]byte(params), &dv.Params)
	_ = json.Unmarshal([]byte(splits), &dv.Splits)
	_ = json.Unmarshal([]byte(stats), &dv.Stats)
	t, _ := time.Parse(time.RFC3339, created)
	dv.CreatedAt = t
	return &dv, nil
}

// LoadManifest reads the manifest file of a version.
func (s *Service) LoadManifest(dv *domain.DatasetVersion) (*Manifest, error) {
	data, err := os.ReadFile(dv.ManifestPath)
	if err != nil {
		return nil, err
	}
	var m Manifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	return &m, nil
}

// Stats computes overall dataset statistics for the UI (class distribution,
// storage, dimension ranges).
func (s *Service) Stats(ctx context.Context, datasetID string) (*domain.DatasetStats, error) {
	st := &domain.DatasetStats{
		ByStatus: map[string]int{}, ByClass: map[string]int{},
	}
	rows, err := s.st.DB().QueryContext(ctx, `SELECT COALESCE(c.name,''), s.status, s.size_bytes, s.width, s.height
          FROM samples s LEFT JOIN classes c ON s.class_id=c.id WHERE s.dataset_id=?`, datasetID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	for rows.Next() {
		var cls, status string
		var size int64
		var w, h int
		if err := rows.Scan(&cls, &status, &size, &w, &h); err != nil {
			return nil, err
		}
		st.TotalSamples++
		st.ByStatus[status]++
		if cls != "" {
			st.ByClass[cls]++
		}
		st.TotalBytes += size
		if w > 0 {
			if st.Widths.Min == 0 || w < st.Widths.Min {
				st.Widths.Min = w
			}
			if w > st.Widths.Max {
				st.Widths.Max = w
			}
		}
		if h > 0 {
			if st.Heights.Min == 0 || h < st.Heights.Min {
				st.Heights.Min = h
			}
			if h > st.Heights.Max {
				st.Heights.Max = h
			}
		}
	}
	if st.TotalSamples > 0 {
		st.AvgBytes = st.TotalBytes / int64(st.TotalSamples)
	}
	return st, rows.Err()
}
