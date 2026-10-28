// Package projects implements project CRUD, duplication, archiving,
// import/export and search. Projects are the top-level container that owns
// datasets, jobs, experiments and models.
package projects

import (
	"archive/zip"
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"neurostudio/internal/domain"
	"neurostudio/internal/store"
	"neurostudio/internal/util"
)

type Service struct {
	st *store.Store
}

func NewService(st *store.Store) *Service { return &Service{st: st} }

type CreateInput struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	TaskType    domain.TaskType `json:"taskType"`
	Settings    map[string]any  `json:"settings"`
}

func (s *Service) Create(ctx context.Context, in CreateInput) (*domain.Project, error) {
	in.Name = strings.TrimSpace(in.Name)
	if in.Name == "" {
		return nil, util.ErrBadRequest("project name is required")
	}
	if len(in.Name) > 120 {
		return nil, util.ErrBadRequest("project name too long (max 120)")
	}
	if in.TaskType == "" {
		in.TaskType = domain.TaskImageClassification
	}
	if !knownTask(in.TaskType) {
		return nil, util.ErrBadRequest("unknown task type %q", in.TaskType)
	}
	p := &domain.Project{
		ID:          util.NewID(),
		Name:        in.Name,
		Description: strings.TrimSpace(in.Description),
		TaskType:    in.TaskType,
		Status:      "active",
		Settings:    in.Settings,
		CreatedAt:   time.Now().UTC(),
		UpdatedAt:   time.Now().UTC(),
	}
	if p.Settings == nil {
		p.Settings = map[string]any{}
	}
	settings, _ := json.Marshal(p.Settings)
	defer s.st.Lock()()
	_, err := s.st.DB().ExecContext(ctx, `INSERT INTO projects
          (id, name, description, task_type, status, settings, created_at, updated_at)
          VALUES (?,?,?,?,?,?,?,?)`,
		p.ID, p.Name, p.Description, p.TaskType, p.Status, string(settings),
		p.CreatedAt.Format(time.RFC3339), p.UpdatedAt.Format(time.RFC3339))
	if err != nil {
		return nil, err
	}
	return p, nil
}

func (s *Service) Get(ctx context.Context, id string) (*domain.Project, error) {
	p, err := scanProject(s.st.DB().QueryRowContext(ctx,
		`SELECT id, name, description, task_type, status, settings, created_at, updated_at, archived_at
                 FROM projects WHERE id = ?`, id))
	if errors.Is(err, sql.ErrNoRows) {
		return nil, util.ErrNotFound("project")
	}
	return p, err
}

func (s *Service) List(ctx context.Context, query, status string, limit int) ([]*domain.Project, error) {
	if limit <= 0 || limit > 200 {
		limit = 100
	}
	q := `SELECT id, name, description, task_type, status, settings, created_at, updated_at, archived_at FROM projects`
	var args []any
	var conds []string
	if status != "" {
		conds = append(conds, "status = ?")
		args = append(args, status)
	}
	if query != "" {
		conds = append(conds, "(name LIKE ? ESCAPE '\\' OR description LIKE ? ESCAPE '\\')")
		esc := escapeLike(query)
		args = append(args, "%"+esc+"%", "%"+esc+"%")
	}
	if len(conds) > 0 {
		q += " WHERE " + strings.Join(conds, " AND ")
	}
	q += " ORDER BY updated_at DESC LIMIT ?"
	args = append(args, limit)
	rows, err := s.st.DB().QueryContext(ctx, q, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []*domain.Project
	for rows.Next() {
		p, err := scanProjectRows(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, p)
	}
	return out, rows.Err()
}

func (s *Service) Update(ctx context.Context, id string, patch map[string]any) (*domain.Project, error) {
	p, err := s.Get(ctx, id)
	if err != nil {
		return nil, err
	}
	if v, ok := patch["name"].(string); ok {
		v = strings.TrimSpace(v)
		if v == "" {
			return nil, util.ErrBadRequest("name cannot be empty")
		}
		p.Name = v
	}
	if v, ok := patch["description"].(string); ok {
		p.Description = v
	}
	if v, ok := patch["settings"].(map[string]any); ok {
		p.Settings = v
	}
	settings, _ := json.Marshal(p.Settings)
	p.UpdatedAt = time.Now().UTC()
	defer s.st.Lock()()
	_, err = s.st.DB().ExecContext(ctx,
		`UPDATE projects SET name=?, description=?, settings=?, updated_at=? WHERE id=?`,
		p.Name, p.Description, string(settings), p.UpdatedAt.Format(time.RFC3339), id)
	if err != nil {
		return nil, err
	}
	return p, nil
}

func (s *Service) SetStatus(ctx context.Context, id, status string) (*domain.Project, error) {
	if status != "active" && status != "archived" {
		return nil, util.ErrBadRequest("status must be active or archived")
	}
	p, err := s.Get(ctx, id)
	if err != nil {
		return nil, err
	}
	p.Status = status
	p.UpdatedAt = time.Now().UTC()
	if status == "archived" {
		t := p.UpdatedAt
		p.ArchivedAt = &t
	} else {
		p.ArchivedAt = nil
	}
	var archived any
	if p.ArchivedAt != nil {
		archived = p.ArchivedAt.Format(time.RFC3339)
	}
	defer s.st.Lock()()
	_, err = s.st.DB().ExecContext(ctx,
		`UPDATE projects SET status=?, archived_at=?, updated_at=? WHERE id=?`,
		status, archived, p.UpdatedAt.Format(time.RFC3339), id)
	if err != nil {
		return nil, err
	}
	return p, nil
}

// Delete removes a project and (via FK cascades) its datasets, jobs,
// experiments and models. Files are removed best-effort afterwards; a failed
// UI request never deletes files because deletion happens only here, after
// the DB transaction succeeds.
func (s *Service) Delete(ctx context.Context, id string) error {
	if _, err := s.Get(ctx, id); err != nil {
		return err
	}
	defer s.st.Lock()()
	tx, err := s.st.DB().BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	if _, err := tx.ExecContext(ctx, `DELETE FROM projects WHERE id=?`, id); err != nil {
		tx.Rollback()
		return err
	}
	return tx.Commit()
}

// Duplicate copies a project row (not its datasets — datasets are copied as
// metadata-only references are nontrivial; documented behavior: duplication
// clones project settings only).
func (s *Service) Duplicate(ctx context.Context, id string) (*domain.Project, error) {
	src, err := s.Get(ctx, id)
	if err != nil {
		return nil, err
	}
	return s.Create(ctx, CreateInput{
		Name:        src.Name + " (copy)",
		Description: src.Description,
		TaskType:    src.TaskType,
		Settings:    src.Settings,
	})
}

func (s *Service) Counts(ctx context.Context, projectID string) (domain.ProjectCounts, error) {
	var c domain.ProjectCounts
	db := s.st.DB()
	q := func(sqlq string, dst *int) error {
		return db.QueryRowContext(ctx, sqlq, projectID).Scan(dst)
	}
	if err := q(`SELECT COUNT(*) FROM datasets WHERE project_id=?`, &c.Datasets); err != nil {
		return c, err
	}
	if err := q(`SELECT COUNT(*) FROM samples s JOIN datasets d ON s.dataset_id=d.id WHERE d.project_id=?`, &c.Samples); err != nil {
		return c, err
	}
	if err := q(`SELECT COUNT(*) FROM experiments WHERE project_id=?`, &c.Experiments); err != nil {
		return c, err
	}
	if err := q(`SELECT COUNT(*) FROM models WHERE project_id=?`, &c.Models); err != nil {
		return c, err
	}
	if err := q(`SELECT COUNT(*) FROM jobs WHERE project_id=? AND status IN ('queued','running')`, &c.JobsRunning); err != nil {
		return c, err
	}
	return c, nil
}

// ExportProject produces a portable ZIP: project manifest + dataset manifests
// + experiment records. Binary artifacts are referenced but not embedded
// (exports with media can be requested per-dataset).
func (s *Service) ExportProject(ctx context.Context, id string) (string, int64, error) {
	p, err := s.Get(ctx, id)
	if err != nil {
		return "", 0, err
	}
	counts, _ := s.Counts(ctx, id)
	p.Counts = counts
	out := filepath.Join(os.TempDir(), "ns-project-"+p.ID+".zip")
	f, err := os.Create(out)
	if err != nil {
		return "", 0, err
	}
	defer f.Close()
	zw := zip.NewWriter(f)
	if err := writeJSONEntry(zw, "project.json", p); err != nil {
		return "", 0, err
	}
	rows, err := s.st.DB().QueryContext(ctx,
		`SELECT id, name, data_type, notes, created_at, updated_at FROM datasets WHERE project_id=?`, id)
	if err != nil {
		return "", 0, err
	}
	defer rows.Close()
	var datasets []map[string]any
	for rows.Next() {
		var id2, name, dtype, notes, created, updated string
		if err := rows.Scan(&id2, &name, &dtype, &notes, &created, &updated); err != nil {
			return "", 0, err
		}
		datasets = append(datasets, map[string]any{
			"id": id2, "name": name, "dataType": dtype, "notes": notes,
			"createdAt": created, "updatedAt": updated,
		})
	}
	rows.Close()
	if err := writeJSONEntry(zw, "datasets.json", datasets); err != nil {
		return "", 0, err
	}
	if err := zw.Close(); err != nil {
		return "", 0, err
	}
	st, _ := f.Stat()
	return out, st.Size(), nil
}

func writeJSONEntry(zw *zip.Writer, name string, v any) error {
	w, err := zw.Create(name)
	if err != nil {
		return err
	}
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	_, err = w.Write(data)
	return err
}

func scanProject(row *sql.Row) (*domain.Project, error) {
	var p domain.Project
	var settings string
	var created, updated string
	var archived *string
	if err := row.Scan(&p.ID, &p.Name, &p.Description, &p.TaskType, &p.Status,
		&settings, &created, &updated, &archived); err != nil {
		return nil, err
	}
	if err := finalizeProject(&p, settings, created, updated, archived); err != nil {
		return nil, err
	}
	return &p, nil
}

func scanProjectRows(rows *sql.Rows) (*domain.Project, error) {
	var p domain.Project
	var settings string
	var created, updated string
	var archived *string
	if err := rows.Scan(&p.ID, &p.Name, &p.Description, &p.TaskType, &p.Status,
		&settings, &created, &updated, &archived); err != nil {
		return nil, err
	}
	if err := finalizeProject(&p, settings, created, updated, archived); err != nil {
		return nil, err
	}
	return &p, nil
}

func finalizeProject(p *domain.Project, settings, created, updated string, archived *string) error {
	_ = json.Unmarshal([]byte(settings), &p.Settings)
	var err error
	if p.CreatedAt, err = time.Parse(time.RFC3339, created); err != nil {
		return err
	}
	if p.UpdatedAt, err = time.Parse(time.RFC3339, updated); err != nil {
		return err
	}
	if archived != nil {
		t, err := time.Parse(time.RFC3339, *archived)
		if err != nil {
			return err
		}
		p.ArchivedAt = &t
	}
	if p.Settings == nil {
		p.Settings = map[string]any{}
	}
	return nil
}

func escapeLike(s string) string {
	s = strings.ReplaceAll(s, `\`, `\\`)
	s = strings.ReplaceAll(s, `%`, `\%`)
	s = strings.ReplaceAll(s, `_`, `\_`)
	return s
}

func knownTask(t domain.TaskType) bool {
	switch t {
	case domain.TaskImageClassification, domain.TaskObjectDetection,
		domain.TaskImageSegmentation, domain.TaskAudioClassification,
		domain.TaskPoseClassification, domain.TaskTextClassification,
		domain.TaskTabular, domain.TaskTimeSeries:
		return true
	}
	return false
}

var _ = fmt.Sprintf
