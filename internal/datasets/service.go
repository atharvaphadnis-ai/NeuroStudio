// Package datasets implements dataset management: creation, class and
// sample CRUD, robust import (multipart/zip/directory), content-hash
// duplicate detection, corrupt-file detection, seeded reproducible splits,
// dataset versioning with manifests, statistics and exports.
package datasets

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"neurostudio/internal/domain"
	"neurostudio/internal/store"
	"neurostudio/internal/util"
)

type Service struct {
	st         *store.Store
	dataRoot   string                   // datasets root: <data>/datasets
	validators map[domain.DataType]bool // data types with real pipelines
}

func NewService(st *store.Store, datasetsRoot string) *Service {
	return &Service{st: st, dataRoot: datasetsRoot}
}

// SupportedDataTypes returns which dataset types have storage/preview support.
// Trainability is a separate concept (see TrainingStudio).
func SupportedDataTypes() []domain.DataType {
	return []domain.DataType{
		domain.DataImage, domain.DataAudio, domain.DataVideo,
		domain.DataTabular, domain.DataTypeText, domain.DataSeries,
	}
}

// TrainableTypes reports data types for which a genuine training pipeline
// exists. Everything else is stored and previewable but never advertised as
// trainable.
func TrainableTypes() map[domain.DataType]bool {
	return map[domain.DataType]bool{domain.DataImage: true}
}

type CreateInput struct {
	Name     string          `json:"name"`
	DataType domain.DataType `json:"dataType"`
	Notes    string          `json:"notes"`
}

func (s *Service) Create(ctx context.Context, projectID string, in CreateInput) (*domain.Dataset, error) {
	in.Name = strings.TrimSpace(in.Name)
	if in.Name == "" {
		return nil, util.ErrBadRequest("dataset name is required")
	}
	if in.DataType == "" {
		in.DataType = domain.DataImage
	}
	ok := false
	for _, t := range SupportedDataTypes() {
		if t == in.DataType {
			ok = true
		}
	}
	if !ok {
		return nil, util.ErrBadRequest("unsupported data type %q", in.DataType)
	}
	if _, err := s.st.DB().ExecContext(ctx,
		`SELECT 1 FROM projects WHERE id=?`, projectID); err != nil {
		return nil, util.ErrNotFound("project")
	}
	ds := &domain.Dataset{
		ID: util.NewID(), ProjectID: projectID, Name: in.Name,
		DataType: in.DataType, Notes: in.Notes,
		CreatedAt: time.Now().UTC(), UpdatedAt: time.Now().UTC(),
	}
	defer s.st.Lock()()
	_, err := s.st.DB().ExecContext(ctx, `INSERT INTO datasets
	  (id, project_id, name, data_type, notes, created_at, updated_at) VALUES (?,?,?,?,?,?,?)`,
		ds.ID, ds.ProjectID, ds.Name, ds.DataType, ds.Notes,
		ds.CreatedAt.Format(time.RFC3339), ds.UpdatedAt.Format(time.RFC3339))
	if err != nil {
		return nil, err
	}
	return ds, nil
}

func (s *Service) List(ctx context.Context, projectID string) ([]*domain.Dataset, error) {
	rows, err := s.st.DB().QueryContext(ctx, `SELECT id, project_id, name, data_type, notes, created_at, updated_at
	  FROM datasets WHERE project_id=? ORDER BY created_at DESC`, projectID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []*domain.Dataset{}
	for rows.Next() {
		var d domain.Dataset
		var created, updated string
		if err := rows.Scan(&d.ID, &d.ProjectID, &d.Name, &d.DataType, &d.Notes, &created, &updated); err != nil {
			return nil, err
		}
		if d.CreatedAt, err = time.Parse(time.RFC3339, created); err != nil {
			return nil, err
		}
		if d.UpdatedAt, err = time.Parse(time.RFC3339, updated); err != nil {
			return nil, err
		}
		out = append(out, &d)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	rows.Close()
	// Enrich after the outer rows are drained: the store allows a single
	// connection, so nested queries inside an open iterator would deadlock.
	for _, d := range out {
		d.ClassCount, _ = s.countClasses(d.ID)
		d.SampleCount, _ = s.countSamples(d.ID)
	}
	return out, nil
}

func (s *Service) Get(ctx context.Context, id string) (*domain.Dataset, error) {
	var d domain.Dataset
	var created, updated string
	err := s.st.DB().QueryRowContext(ctx, `SELECT id, project_id, name, data_type, notes, created_at, updated_at
	  FROM datasets WHERE id=?`, id).
		Scan(&d.ID, &d.ProjectID, &d.Name, &d.DataType, &d.Notes, &created, &updated)
	if err == sql.ErrNoRows {
		return nil, util.ErrNotFound("dataset")
	}
	if err != nil {
		return nil, err
	}
	var e1, e2 error
	if d.CreatedAt, e1 = time.Parse(time.RFC3339, created); e1 != nil {
		return nil, e1
	}
	if d.UpdatedAt, e2 = time.Parse(time.RFC3339, updated); e2 != nil {
		return nil, e2
	}
	d.ClassCount, _ = s.countClasses(id)
	d.SampleCount, _ = s.countSamples(id)
	d.Classes, _ = s.ListClasses(ctx, id)
	return &d, nil
}

func (s *Service) Delete(ctx context.Context, id string) error {
	if _, err := s.Get(ctx, id); err != nil {
		return err
	}
	defer s.st.Lock()()
	_, err := s.st.DB().ExecContext(ctx, `DELETE FROM datasets WHERE id=?`, id)
	return err
}

// ---- classes ----

var classColors = []string{
	"#6366f1", "#10b981", "#f59e0b", "#ef4444", "#06b6d4",
	"#8b5cf6", "#ec4899", "#84cc16", "#f97316", "#14b8a6",
}

func (s *Service) AddClass(ctx context.Context, datasetID, name string) (*domain.Class, error) {
	name = strings.TrimSpace(name)
	if name == "" {
		return nil, util.ErrBadRequest("class name required")
	}
	if len(name) > 80 {
		return nil, util.ErrBadRequest("class name too long")
	}
	var ord int
	if err := s.st.DB().QueryRowContext(ctx,
		`SELECT COALESCE(MAX(ord),-1)+1 FROM classes WHERE dataset_id=?`, datasetID).Scan(&ord); err != nil {
		return nil, err
	}
	c := &domain.Class{
		ID: util.NewID(), DatasetID: datasetID, Name: name,
		Color: classColors[ord%len(classColors)], Ord: ord,
		CreatedAt: time.Now().UTC(),
	}
	defer s.st.Lock()()
	_, err := s.st.DB().ExecContext(ctx, `INSERT INTO classes (id, dataset_id, name, color, ord, created_at)
	  VALUES (?,?,?,?,?,?)`, c.ID, c.DatasetID, c.Name, c.Color, c.Ord,
		c.CreatedAt.Format(time.RFC3339))
	if err != nil {
		if strings.Contains(err.Error(), "UNIQUE") {
			return nil, util.Err(409, "duplicate_class", "class name already exists")
		}
		return nil, err
	}
	return c, nil
}

func (s *Service) ListClasses(ctx context.Context, datasetID string) ([]domain.Class, error) {
	rows, err := s.st.DB().QueryContext(ctx, `SELECT id, dataset_id, name, color, ord, created_at
	  FROM classes WHERE dataset_id=? ORDER BY ord`, datasetID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []domain.Class{}
	for rows.Next() {
		var c domain.Class
		var created string
		if err := rows.Scan(&c.ID, &c.DatasetID, &c.Name, &c.Color, &c.Ord, &created); err != nil {
			return nil, err
		}
		c.CreatedAt, _ = time.Parse(time.RFC3339, created)
		out = append(out, c)
	}
	return out, rows.Err()
}

func (s *Service) RenameClass(ctx context.Context, classID, name string) (*domain.Class, error) {
	name = strings.TrimSpace(name)
	if name == "" {
		return nil, util.ErrBadRequest("class name required")
	}
	defer s.st.Lock()()
	res, err := s.st.DB().ExecContext(ctx, `UPDATE classes SET name=? WHERE id=?`, name, classID)
	if err != nil {
		return nil, err
	}
	if n, _ := res.RowsAffected(); n == 0 {
		return nil, util.ErrNotFound("class")
	}
	var c domain.Class
	var created string
	err = s.st.DB().QueryRowContext(ctx,
		`SELECT id, dataset_id, name, color, ord, created_at FROM classes WHERE id=?`, classID).
		Scan(&c.ID, &c.DatasetID, &c.Name, &c.Color, &c.Ord, &created)
	return &c, err
}

func (s *Service) DeleteClass(ctx context.Context, classID string) error {
	defer s.st.Lock()()
	res, err := s.st.DB().ExecContext(ctx, `DELETE FROM classes WHERE id=?`, classID)
	if err != nil {
		return err
	}
	if n, _ := res.RowsAffected(); n == 0 {
		return util.ErrNotFound("class")
	}
	return nil
}

// ---- samples ----

func (s *Service) countClasses(datasetID string) (int, error) {
	var n int
	err := s.st.DB().QueryRow(`SELECT COUNT(*) FROM classes WHERE dataset_id=?`, datasetID).Scan(&n)
	return n, err
}

func (s *Service) countSamples(datasetID string) (int, error) {
	var n int
	err := s.st.DB().QueryRow(`SELECT COUNT(*) FROM samples WHERE dataset_id=?`, datasetID).Scan(&n)
	return n, err
}

func (s *Service) SampleFile(datasetID, relPath string) (string, error) {
	return util.SafeJoin(s.dataRoot+"/"+datasetID, relPath)
}

func (s *Service) DataRoot() string { return s.dataRoot }

// ListSamples returns a page of samples with class names joined.
func (s *Service) ListSamples(ctx context.Context, datasetID, classID, status string, page, pageSize int) ([]domain.Sample, int, error) {
	if pageSize <= 0 || pageSize > 500 {
		pageSize = 60
	}
	if page < 0 {
		page = 0
	}
	q := `SELECT s.id, s.dataset_id, s.class_id, COALESCE(c.name,''), s.filename, s.rel_path,
	  s.media_type, s.size_bytes, s.sha256, s.width, s.height, s.duration_ms, s.status, s.meta, s.created_at
	  FROM samples s LEFT JOIN classes c ON s.class_id=c.id WHERE s.dataset_id=?`
	var args []any
	args = append(args, datasetID)
	if classID != "" {
		q += ` AND s.class_id=?`
		args = append(args, classID)
	}
	if status != "" {
		q += ` AND s.status=?`
		args = append(args, status)
	}
	var total int
	if err := s.st.DB().QueryRowContext(ctx,
		`SELECT COUNT(*) FROM samples s WHERE s.dataset_id=?`+
			whereSuffix(classID, status), args...).Scan(&total); err != nil {
		return nil, 0, err
	}
	q += ` ORDER BY s.created_at, s.id LIMIT ? OFFSET ?`
	args = append(args, pageSize, page*pageSize)
	rows, err := s.st.DB().QueryContext(ctx, q, args...)
	if err != nil {
		return nil, 0, err
	}
	defer rows.Close()
	out := []domain.Sample{}
	for rows.Next() {
		sample, err := scanSample(rows)
		if err != nil {
			return nil, 0, err
		}
		out = append(out, *sample)
	}
	return out, total, rows.Err()
}

func whereSuffix(classID, status string) string {
	s := ""
	if classID != "" {
		s += ` AND s.class_id='` + classID + `'`
	}
	if status != "" {
		s += ` AND s.status='` + status + `'`
	}
	return s
}

func scanSample(rows *sql.Rows) (*domain.Sample, error) {
	var sm domain.Sample
	var classID sql.NullString
	var created string
	var meta string
	if err := rows.Scan(&sm.ID, &sm.DatasetID, &classID, &sm.Class, &sm.Filename, &sm.RelPath,
		&sm.MediaType, &sm.SizeBytes, &sm.SHA256, &sm.Width, &sm.Height, &sm.DurationMS,
		&sm.Status, &meta, &created); err != nil {
		return nil, err
	}
	if classID.Valid {
		v := classID.String
		sm.ClassID = &v
	}
	sm.CreatedAt, _ = time.Parse(time.RFC3339, created)
	_ = json.Unmarshal([]byte(meta), &sm.Meta)
	return &sm, nil
}

// Relabel moves a sample to another class (or unassigns with null classID).
func (s *Service) Relabel(ctx context.Context, sampleID string, classID *string) (*domain.Sample, error) {
	defer s.st.Lock()()
	var res sql.Result
	var err error
	if classID == nil {
		res, err = s.st.DB().ExecContext(ctx, `UPDATE samples SET class_id=NULL WHERE id=?`, sampleID)
	} else {
		res, err = s.st.DB().ExecContext(ctx, `UPDATE samples SET class_id=? WHERE id=?`, *classID, sampleID)
	}
	if err != nil {
		return nil, err
	}
	if n, _ := res.RowsAffected(); n == 0 {
		return nil, util.ErrNotFound("sample")
	}
	return s.GetSample(ctx, sampleID)
}

// DeleteSamples removes samples (files remain until storage cleanup; the
// datasets root is reference-counted via sha256 dedupe).
func (s *Service) DeleteSamples(ctx context.Context, ids []string) (int, error) {
	defer s.st.Lock()()
	tx, err := s.st.DB().BeginTx(ctx, nil)
	if err != nil {
		return 0, err
	}
	n := 0
	for _, id := range ids {
		res, err := tx.ExecContext(ctx, `DELETE FROM samples WHERE id=?`, id)
		if err != nil {
			tx.Rollback()
			return n, err
		}
		if r, _ := res.RowsAffected(); r > 0 {
			n++
		}
	}
	return n, tx.Commit()
}

func (s *Service) GetSample(ctx context.Context, sampleID string) (*domain.Sample, error) {
	rows, err := s.st.DB().QueryContext(ctx, `SELECT s.id, s.dataset_id, s.class_id, COALESCE(c.name,''), s.filename, s.rel_path,
	  s.media_type, s.size_bytes, s.sha256, s.width, s.height, s.duration_ms, s.status, s.meta, s.created_at
	  FROM samples s LEFT JOIN classes c ON s.class_id=c.id WHERE s.id=?`, sampleID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	if !rows.Next() {
		return nil, util.ErrNotFound("sample")
	}
	return scanSample(rows)
}

func (s *Service) touchDataset(ctx context.Context, datasetID string) {
	s.st.DB().ExecContext(ctx, `UPDATE datasets SET updated_at=? WHERE id=?`,
		time.Now().UTC().Format(time.RFC3339), datasetID)
}

var _ = fmt.Sprintf
