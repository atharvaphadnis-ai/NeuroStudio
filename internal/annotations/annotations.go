// Package annotations stores image annotations (bounding boxes and labels)
// and exports them in standard formats (COCO JSON, YOLO txt). Annotation
// storage works today; detection *training* arrives with the detection
// backend and is never advertised before then.
package annotations

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"time"

	"neurostudio/internal/store"
	"neurostudio/internal/util"
)

type Service struct {
	st *store.Store
}

func NewService(st *store.Store) *Service { return &Service{st: st} }

type Annotation struct {
	ID        string         `json:"id"`
	SampleID  string         `json:"sampleId"`
	Kind      string         `json:"kind"` // bbox
	Label     string         `json:"label"`
	Data      map[string]any `json:"data"`
	CreatedAt time.Time      `json:"createdAt"`
	UpdatedAt time.Time      `json:"updatedAt"`
}

func (s *Service) Add(ctx context.Context, sampleID, kind, label string, data map[string]any) (*Annotation, error) {
	if kind != "bbox" {
		return nil, util.ErrBadRequest("annotation kind must be 'bbox'")
	}
	if err := validateBBox(data); err != nil {
		return nil, err
	}
	a := &Annotation{
		ID: util.NewID(), SampleID: sampleID, Kind: kind, Label: strings.TrimSpace(label),
		Data: data, CreatedAt: time.Now().UTC(), UpdatedAt: time.Now().UTC(),
	}
	dataJSON, _ := json.Marshal(a.Data)
	defer s.st.Lock()()
	_, err := s.st.DB().ExecContext(ctx, `INSERT INTO annotations (id, sample_id, kind, label, data, created_at, updated_at)
	  VALUES (?,?,?,?,?,?,?)`, a.ID, a.SampleID, a.Kind, a.Label, string(dataJSON),
		a.CreatedAt.Format(time.RFC3339), a.UpdatedAt.Format(time.RFC3339))
	if err != nil {
		return nil, err
	}
	return a, nil
}

func validateBBox(data map[string]any) error {
	for _, k := range []string{"x", "y", "w", "h"} {
		v, ok := data[k].(float64)
		if !ok {
			return util.ErrBadRequest("bbox requires numeric x, y, w, h")
		}
		_ = v
	}
	x, y, w, h := num(data["x"]), num(data["y"]), num(data["w"]), num(data["h"])
	if w <= 0 || h <= 0 {
		return util.ErrBadRequest("bbox w and h must be positive")
	}
	if x < 0 || y < 0 || x+w > 1.0001 || y+h > 1.0001 {
		return util.ErrBadRequest("bbox must stay within the image (normalized 0..1)")
	}
	return nil
}

func num(v any) float64 {
	f, _ := v.(float64)
	return f
}

func (s *Service) List(ctx context.Context, sampleID string) ([]Annotation, error) {
	rows, err := s.st.DB().QueryContext(ctx, `SELECT id, sample_id, kind, label, data, created_at, updated_at
	  FROM annotations WHERE sample_id=? ORDER BY created_at`, sampleID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []Annotation{}
	for rows.Next() {
		var a Annotation
		var data, created, updated string
		if err := rows.Scan(&a.ID, &a.SampleID, &a.Kind, &a.Label, &data, &created, &updated); err != nil {
			return nil, err
		}
		_ = json.Unmarshal([]byte(data), &a.Data)
		a.CreatedAt, _ = time.Parse(time.RFC3339, created)
		a.UpdatedAt, _ = time.Parse(time.RFC3339, updated)
		out = append(out, a)
	}
	return out, rows.Err()
}

func (s *Service) Delete(ctx context.Context, id string) error {
	defer s.st.Lock()()
	res, err := s.st.DB().ExecContext(ctx, `DELETE FROM annotations WHERE id=?`, id)
	if err != nil {
		return err
	}
	if n, _ := res.RowsAffected(); n == 0 {
		return util.ErrNotFound("annotation")
	}
	return nil
}

// sampleMeta is the data needed for export.
type sampleMeta struct {
	SampleID string
	File     string
	Width    int
	Height   int
	Classes  map[string]string // class name -> id
}

// ExportCOCO produces a COCO-style JSON of all bboxes in a dataset.
func (s *Service) ExportCOCO(ctx context.Context, st *store.Store, datasetID string) ([]byte, error) {
	samples, classes, err := datasetContent(ctx, st, datasetID)
	if err != nil {
		return nil, err
	}
	classNameIdx := map[string]int{}
	cats := []map[string]any{}
	for i, c := range classes {
		classNameIdx[c] = i + 1
		cats = append(cats, map[string]any{"id": i + 1, "name": c})
	}
	imgs := []map[string]any{}
	anns := []map[string]any{}
	annID := 1
	for i, sm := range samples {
		imgs = append(imgs, map[string]any{
			"id": i + 1, "file_name": sm.File, "width": sm.Width, "height": sm.Height,
		})
		anns2, err := s.List(ctx, sm.SampleID)
		if err != nil {
			return nil, err
		}
		for _, a := range anns2 {
			if a.Kind != "bbox" {
				continue
			}
			x, y, w, h := num(a.Data["x"]), num(a.Data["y"]), num(a.Data["w"]), num(a.Data["h"])
			anns = append(anns, map[string]any{
				"id": annID, "image_id": i + 1,
				"category_id": classNameIdx[a.Label],
				"bbox":        []float64{x * float64(sm.Width), y * float64(sm.Height), w * float64(sm.Width), h * float64(sm.Height)},
				"area":        w * h * float64(sm.Width) * float64(sm.Height),
				"iscrowd":     0,
			})
			annID++
		}
	}
	out := map[string]any{
		"info":        map[string]any{"description": "NeuroStudio annotation export", "version": "1.0"},
		"images":      imgs,
		"annotations": anns,
		"categories":  cats,
	}
	return json.MarshalIndent(out, "", " ")
}

// ExportYOLO produces a zip-less YOLO bundle description: one .txt per image
// with lines "class_idx cx cy w h" (normalized). Returned as a map of
// filename -> content plus classes.txt.
func (s *Service) ExportYOLO(ctx context.Context, st *store.Store, datasetID string) (map[string]string, error) {
	samples, classes, err := datasetContent(ctx, st, datasetID)
	if err != nil {
		return nil, err
	}
	classIdx := map[string]int{}
	for i, c := range classes {
		classIdx[c] = i
	}
	files := map[string]string{
		"classes.txt": strings.Join(classes, "\n") + "\n",
	}
	counts := map[string]int{}
	for _, sm := range samples {
		anns, err := s.List(ctx, sm.SampleID)
		if err != nil {
			return nil, err
		}
		if len(anns) == 0 {
			continue
		}
		var b strings.Builder
		for _, a := range anns {
			if a.Kind != "bbox" {
				continue
			}
			x, y, w, h := num(a.Data["x"]), num(a.Data["y"]), num(a.Data["w"]), num(a.Data["h"])
			fmt.Fprintf(&b, "%d %.6f %.6f %.6f %.6f\n", classIdx[a.Label], x+w/2, y+h/2, w, h)
		}
		base := sm.File
		if ext := strings.LastIndex(base, "."); ext >= 0 {
			base = base[:ext]
		}
		base = base + ".txt"
		for counts[base] > 0 {
			base = fmt.Sprintf("%s_%d.txt", strings.TrimSuffix(base, ".txt"), counts[base])
		}
		counts[base]++
		files[base] = b.String()
	}
	return files, nil
}

func datasetContent(ctx context.Context, st *store.Store, datasetID string) ([]sampleMeta, []string, error) {
	rows, err := st.DB().QueryContext(ctx, `SELECT id, filename, width, height FROM samples
	  WHERE dataset_id=? AND status='ok' ORDER BY created_at, id`, datasetID)
	if err != nil {
		return nil, nil, err
	}
	defer rows.Close()
	var samples []sampleMeta
	for rows.Next() {
		var sm sampleMeta
		if err := rows.Scan(&sm.SampleID, &sm.File, &sm.Width, &sm.Height); err != nil {
			return nil, nil, err
		}
		if sm.Width == 0 {
			sm.Width = 96
		}
		if sm.Height == 0 {
			sm.Height = 96
		}
		samples = append(samples, sm)
	}
	rows.Close()
	crows, err := st.DB().QueryContext(ctx, `SELECT name FROM classes WHERE dataset_id=? ORDER BY ord`, datasetID)
	if err != nil {
		return nil, nil, err
	}
	defer crows.Close()
	var classes []string
	for crows.Next() {
		var c string
		if err := crows.Scan(&c); err != nil {
			return nil, nil, err
		}
		classes = append(classes, c)
	}
	sort.Strings(classes)
	return samples, classes, crows.Err()
}

var _ = sql.ErrNoRows
var _ = http.StatusOK
