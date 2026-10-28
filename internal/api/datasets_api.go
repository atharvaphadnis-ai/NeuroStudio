package api

import (
	"archive/zip"
	"bytes"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	"image/png"
	_ "image/png"
	"net/http"
	"os"
	"strconv"

	"neurostudio/internal/domain"
	"neurostudio/internal/ml"
	"neurostudio/internal/util"
)

func (s *Server) registerDatasetRoutes(m *http.ServeMux) {
	m.HandleFunc("GET /api/v1/projects/{id}/datasets", s.handleDatasetList)
	m.HandleFunc("POST /api/v1/projects/{id}/datasets", s.handleDatasetCreate)
	m.HandleFunc("POST /api/v1/projects/{id}/sample-dataset", s.handleSampleDataset)
	m.HandleFunc("GET /api/v1/datasets/{id}", s.handleDatasetGet)
	m.HandleFunc("DELETE /api/v1/datasets/{id}", s.handleDatasetDelete)
	m.HandleFunc("GET /api/v1/datasets/{id}/stats", s.handleDatasetStats)
	m.HandleFunc("POST /api/v1/datasets/{id}/classes", s.handleClassCreate)
	m.HandleFunc("PATCH /api/v1/classes/{id}", s.handleClassRename)
	m.HandleFunc("DELETE /api/v1/classes/{id}", s.handleClassDelete)
	m.HandleFunc("POST /api/v1/datasets/{id}/samples/import", s.handleSampleImport)
	m.HandleFunc("POST /api/v1/datasets/{id}/samples/import-zip", s.handleSampleImportZip)
	m.HandleFunc("GET /api/v1/datasets/{id}/samples", s.handleSampleList)
	m.HandleFunc("GET /api/v1/datasets/{id}/versions", s.handleVersionList)
	m.HandleFunc("POST /api/v1/datasets/{id}/versions", s.handleVersionCreate)
	m.HandleFunc("GET /api/v1/datasets/{id}/export/annotations", s.handleAnnotationExport)
	m.HandleFunc("GET /api/v1/datasets/{id}/export/manifest", s.handleManifestExport)
	m.HandleFunc("PATCH /api/v1/samples/{id}", s.handleSampleUpdate)
	m.HandleFunc("POST /api/v1/samples/delete", s.handleSamplesDelete)
	m.HandleFunc("GET /api/v1/samples/{id}/file", s.handleSampleFile)
}

func (s *Server) registerAnnotationRoutes(m *http.ServeMux) {
	m.HandleFunc("POST /api/v1/annotations", s.handleAnnotationCreate)
	m.HandleFunc("GET /api/v1/samples/{id}/annotations", s.handleAnnotationList)
	m.HandleFunc("DELETE /api/v1/annotations/{id}", s.handleAnnotationDelete)
}

func (s *Server) handleDatasetList(w http.ResponseWriter, r *http.Request) {
	list, err := s.Datasets.List(r.Context(), r.PathValue("id"))
	if err != nil {
		util.WriteErr(w, err)
		return
	}
	util.WriteJSON(w, 200, list)
}

func (s *Server) handleDatasetCreate(w http.ResponseWriter, r *http.Request) {
	var in struct {
		Name     string          `json:"name"`
		DataType domain.DataType `json:"dataType"`
		Notes    string          `json:"notes"`
	}
	if err := util.DecodeJSON(r, &in); err != nil {
		util.WriteErr(w, err)
		return
	}
	ds, err := s.Datasets.Create(r.Context(), r.PathValue("id"), in)
	if err != nil {
		util.WriteErr(w, err)
		return
	}
	util.WriteJSON(w, 201, ds)
}

func (s *Server) handleSampleDataset(w http.ResponseWriter, r *http.Request) {
	var in struct {
		Name     string `json:"name"`
		Classes  int    `json:"classes"`
		PerClass int    `json:"perClass"`
		Seed     int64  `json:"seed"`
	}
	if err := util.DecodeJSON(r, &in); err != nil {
		util.WriteErr(w, err)
		return
	}
	if in.Name == "" {
		in.Name = "Sample Shapes"
	}
	if in.Seed == 0 {
		in.Seed = 42
	}
	ds, res, err := s.Datasets.GenerateSampleDataset(r.Context(), r.PathValue("id"), in.Name, in.Classes, in.PerClass, in.Seed)
	if err != nil {
		util.WriteErr(w, err)
		return
	}
	util.WriteJSON(w, 201, map[string]any{"dataset": ds, "result": res})
}

func (s *Server) handleDatasetGet(w http.ResponseWriter, r *http.Request) {
	ds, err := s.Datasets.Get(r.Context(), r.PathValue("id"))
	if err != nil {
		util.WriteErr(w, err)
		return
	}
	stats, _ := s.Datasets.Stats(r.Context(), ds.ID)
	ds.Stats = stats
	util.WriteJSON(w, 200, ds)
}

func (s *Server) handleDatasetDelete(w http.ResponseWriter, r *http.Request) {
	if r.URL.Query().Get("confirm") != "true" {
		util.WriteErr(w, util.Err(400, "confirm_required", "destructive operation: add ?confirm=true"))
		return
	}
	if err := s.Datasets.Delete(r.Context(), r.PathValue("id")); err != nil {
		util.WriteErr(w, err)
		return
	}
	util.WriteJSON(w, 200, map[string]any{"deleted": true})
}

func (s *Server) handleDatasetStats(w http.ResponseWriter, r *http.Request) {
	st, err := s.Datasets.Stats(r.Context(), r.PathValue("id"))
	if err != nil {
		util.WriteErr(w, err)
		return
	}
	util.WriteJSON(w, 200, st)
}

func (s *Server) handleClassCreate(w http.ResponseWriter, r *http.Request) {
	var in struct {
		Name string `json:"name"`
	}
	if err := util.DecodeJSON(r, &in); err != nil {
		util.WriteErr(w, err)
		return
	}
	c, err := s.Datasets.AddClass(r.Context(), r.PathValue("id"), in.Name)
	if err != nil {
		util.WriteErr(w, err)
		return
	}
	util.WriteJSON(w, 201, c)
}

func (s *Server) handleClassRename(w http.ResponseWriter, r *http.Request) {
	var in struct {
		Name string `json:"name"`
	}
	if err := util.DecodeJSON(r, &in); err != nil {
		util.WriteErr(w, err)
		return
	}
	c, err := s.Datasets.RenameClass(r.Context(), r.PathValue("id"), in.Name)
	if err != nil {
		util.WriteErr(w, err)
		return
	}
	util.WriteJSON(w, 200, c)
}

func (s *Server) handleClassDelete(w http.ResponseWriter, r *http.Request) {
	if err := s.Datasets.DeleteClass(r.Context(), r.PathValue("id")); err != nil {
		util.WriteErr(w, err)
		return
	}
	util.WriteJSON(w, 200, map[string]any{"deleted": true})
}

func (s *Server) handleSampleImport(w http.ResponseWriter, r *http.Request) {
	ds, err := s.Datasets.Get(r.Context(), r.PathValue("id"))
	if err != nil {
		util.WriteErr(w, err)
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, s.App.Config.MaxUploadMB<<20)
	res, err := s.Datasets.ImportMultipart(r.Context(), ds, r, r.URL.Query().Get("class"))
	if err != nil {
		util.WriteErr(w, err)
		return
	}
	util.WriteJSON(w, 200, res)
}

func (s *Server) handleSampleImportZip(w http.ResponseWriter, r *http.Request) {
	ds, err := s.Datasets.Get(r.Context(), r.PathValue("id"))
	if err != nil {
		util.WriteErr(w, err)
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, s.App.Config.MaxUploadMB<<20)
	res, err := s.Datasets.ImportZIP(r.Context(), ds, r.Body, s.App.Config.MaxUploadMB<<20)
	if err != nil {
		util.WriteErr(w, err)
		return
	}
	util.WriteJSON(w, 200, res)
}

func (s *Server) handleSampleList(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	page, _ := strconv.Atoi(q.Get("page"))
	size, _ := strconv.Atoi(q.Get("pageSize"))
	samples, total, err := s.Datasets.ListSamples(r.Context(), r.PathValue("id"),
		q.Get("classId"), q.Get("status"), page, size)
	if err != nil {
		util.WriteErr(w, err)
		return
	}
	if samples == nil {
		samples = []domain.Sample{}
	}
	util.WriteJSON(w, 200, map[string]any{"samples": samples, "total": total, "page": page})
}

func (s *Server) handleSampleUpdate(w http.ResponseWriter, r *http.Request) {
	var in struct {
		ClassID *string `json:"classId"`
	}
	if err := util.DecodeJSON(r, &in); err != nil {
		util.WriteErr(w, err)
		return
	}
	sm, err := s.Datasets.Relabel(r.Context(), r.PathValue("id"), in.ClassID)
	if err != nil {
		util.WriteErr(w, err)
		return
	}
	util.WriteJSON(w, 200, sm)
}

func (s *Server) handleSamplesDelete(w http.ResponseWriter, r *http.Request) {
	var in struct {
		IDs []string `json:"ids"`
	}
	if err := util.DecodeJSON(r, &in); err != nil {
		util.WriteErr(w, err)
		return
	}
	if len(in.IDs) == 0 || len(in.IDs) > 5000 {
		util.WriteErr(w, util.ErrBadRequest("ids must contain 1..5000 entries"))
		return
	}
	n, err := s.Datasets.DeleteSamples(r.Context(), in.IDs)
	if err != nil {
		util.WriteErr(w, err)
		return
	}
	util.WriteJSON(w, 200, map[string]any{"deleted": n})
}

// handleSampleFile streams the original file or a resized thumbnail
// (?size=NN for images).
func (s *Server) handleSampleFile(w http.ResponseWriter, r *http.Request) {
	sm, err := s.Datasets.GetSample(r.Context(), r.PathValue("id"))
	if err != nil {
		util.WriteErr(w, err)
		return
	}
	path, err := s.Datasets.SampleFile(sm.DatasetID, sm.RelPath)
	if err != nil {
		util.WriteErr(w, err)
		return
	}
	data, err := os.ReadFile(path)
	if err != nil {
		util.WriteErr(w, util.ErrNotFound("sample file"))
		return
	}
	w.Header().Set("Content-Type", sm.MediaType)
	w.Header().Set("Cache-Control", "private, max-age=86400")
	if size := atoiOr(r.URL.Query().Get("size"), 0); size > 0 && size <= 512 && isImage(sm.MediaType) {
		img, _, err := image.Decode(bytes.NewReader(data))
		if err != nil {
			util.WriteErr(w, util.Err(500, "decode_failed", "cannot decode image"))
			return
		}
		w2, h2 := fitRect(img.Bounds().Dx(), img.Bounds().Dy(), size)
		thumb := ml.ResizeBilinear(img, w2, h2)
		var buf bytes.Buffer
		if err := png.Encode(&buf, thumb); err != nil {
			util.WriteErr(w, err)
			return
		}
		w.Header().Set("Content-Type", "image/png")
		_, _ = w.Write(buf.Bytes())
		return
	}
	w.Header().Set("Content-Length", strconv.FormatInt(int64(len(data)), 10))
	_, _ = w.Write(data)
}

func fitRect(w, h, max int) (int, int) {
	if w <= max && h <= max {
		return w, h
	}
	if w >= h {
		return max, max * h / w
	}
	return max * w / h, max
}

func isImage(mt string) bool {
	return len(mt) >= 6 && mt[:6] == "image/"
}

// ---- versions ----

func (s *Server) handleVersionList(w http.ResponseWriter, r *http.Request) {
	list, err := s.Datasets.ListVersions(r.Context(), r.PathValue("id"))
	if err != nil {
		util.WriteErr(w, err)
		return
	}
	if list == nil {
		list = []*domain.DatasetVersion{}
	}
	util.WriteJSON(w, 200, list)
}

func (s *Server) handleVersionCreate(w http.ResponseWriter, r *http.Request) {
	var in struct {
		Train float64 `json:"train"`
		Val   float64 `json:"val"`
		Test  float64 `json:"test"`
		Seed  int64   `json:"seed"`
	}
	if err := util.DecodeJSON(r, &in); err != nil {
		util.WriteErr(w, err)
		return
	}
	if in.Seed == 0 {
		in.Seed = 42
	}
	dv, err := s.Datasets.CreateVersion(r.Context(), r.PathValue("id"),
		domain.SplitParams{Train: in.Train, Val: in.Val, Test: in.Test, Seed: in.Seed})
	if err != nil {
		util.WriteErr(w, err)
		return
	}
	dv.Splits = nil
	util.WriteJSON(w, 201, dv)
}

func (s *Server) handleManifestExport(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	dv, err := s.Datasets.GetVersion(r.Context(), r.URL.Query().Get("versionId"))
	if err != nil || dv.DatasetID != id {
		versions, verr := s.Datasets.ListVersions(r.Context(), id)
		if verr != nil || len(versions) == 0 {
			util.WriteErr(w, util.ErrNotFound("dataset version"))
			return
		}
		dv = versions[0]
	}
	m, err := s.Datasets.LoadManifest(dv)
	if err != nil {
		util.WriteErr(w, err)
		return
	}
	util.WriteJSON(w, 200, m)
}

// ---- annotations ----

func (s *Server) handleAnnotationCreate(w http.ResponseWriter, r *http.Request) {
	var in struct {
		SampleID string         `json:"sampleId"`
		Kind     string         `json:"kind"`
		Label    string         `json:"label"`
		Data     map[string]any `json:"data"`
	}
	if err := util.DecodeJSON(r, &in); err != nil {
		util.WriteErr(w, err)
		return
	}
	a, err := s.Annotations.Add(r.Context(), in.SampleID, in.Kind, in.Label, in.Data)
	if err != nil {
		util.WriteErr(w, err)
		return
	}
	util.WriteJSON(w, 201, a)
}

func (s *Server) handleAnnotationList(w http.ResponseWriter, r *http.Request) {
	list, err := s.Annotations.List(r.Context(), r.PathValue("id"))
	if err != nil {
		util.WriteErr(w, err)
		return
	}
	util.WriteJSON(w, 200, list)
}

func (s *Server) handleAnnotationDelete(w http.ResponseWriter, r *http.Request) {
	if err := s.Annotations.Delete(r.Context(), r.PathValue("id")); err != nil {
		util.WriteErr(w, err)
		return
	}
	util.WriteJSON(w, 200, map[string]any{"deleted": true})
}

func (s *Server) handleAnnotationExport(w http.ResponseWriter, r *http.Request) {
	datasetID := r.PathValue("id")
	format := r.URL.Query().Get("format")
	switch format {
	case "coco", "":
		data, err := s.Annotations.ExportCOCO(r.Context(), s.Store, datasetID)
		if err != nil {
			util.WriteErr(w, err)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Content-Disposition", "attachment; filename=\"annotations-coco.json\"")
		_, _ = w.Write(data)
	case "yolo":
		files, err := s.Annotations.ExportYOLO(r.Context(), s.Store, datasetID)
		if err != nil {
			util.WriteErr(w, err)
			return
		}
		var buf bytes.Buffer
		zw := zip.NewWriter(&buf)
		for name, content := range files {
			fe, err := zw.Create("labels/" + name)
			if err != nil {
				util.WriteErr(w, err)
				return
			}
			_, _ = fe.Write([]byte(content))
		}
		zw.Close()
		w.Header().Set("Content-Type", "application/zip")
		w.Header().Set("Content-Disposition", "attachment; filename=\"annotations-yolo.zip\"")
		_, _ = w.Write(buf.Bytes())
	default:
		util.WriteErr(w, util.ErrBadRequest("format must be coco or yolo"))
	}
}
