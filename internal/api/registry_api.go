package api

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"neurostudio/internal/util"
)

// registerRegistryRoutes exposes the experiment log and the model registry.
// Replaces the Stage 1 stub.
func (s *Server) registerRegistryRoutes(m *http.ServeMux) {
	m.HandleFunc("GET /api/v1/projects/{id}/experiments", s.handleExperimentList)
	m.HandleFunc("GET /api/v1/experiments/{id}", s.handleExperimentGet)
	m.HandleFunc("DELETE /api/v1/experiments/{id}", s.handleExperimentDelete)
	m.HandleFunc("GET /api/v1/projects/{id}/models", s.handleModelList)
	m.HandleFunc("GET /api/v1/models/{id}", s.handleModelGet)
	m.HandleFunc("PATCH /api/v1/models/{id}", s.handleModelUpdate)
	m.HandleFunc("DELETE /api/v1/models/{id}", s.handleModelDelete)
	m.HandleFunc("GET /api/v1/models/{id}/versions", s.handleModelVersionList)
	m.HandleFunc("GET /api/v1/models/{id}/versions/{vid}", s.handleModelVersionGet)
	m.HandleFunc("GET /api/v1/models/{id}/versions/{vid}/download", s.handleModelDownload)
}

// ---- experiments ----

func (s *Server) handleExperimentList(w http.ResponseWriter, r *http.Request) {
	rows, err := s.Store.DB().QueryContext(r.Context(), `SELECT id, job_id, name, task_type, backend_id,
          architecture, seed, config, dataset_id, dataset_version_id, metrics, curves, artifacts, runtime, notes, created_at
          FROM experiments WHERE project_id=? ORDER BY created_at DESC LIMIT 200`, r.PathValue("id"))
	if err != nil {
		util.WriteErr(w, err)
		return
	}
	defer rows.Close()
	out := []map[string]any{}
	for rows.Next() {
		e, err := scanExperiment(rows)
		if err != nil {
			util.WriteErr(w, err)
			return
		}
		out = append(out, e)
	}
	util.WriteJSON(w, 200, out)
}

func (s *Server) handleExperimentGet(w http.ResponseWriter, r *http.Request) {
	row := s.Store.DB().QueryRowContext(r.Context(), `SELECT id, job_id, name, task_type, backend_id,
          architecture, seed, config, dataset_id, dataset_version_id, metrics, curves, artifacts, runtime, notes, created_at
          FROM experiments WHERE id=?`, r.PathValue("id"))
	e, err := scanExperiment(row)
	if err == sql.ErrNoRows {
		util.WriteErr(w, util.ErrNotFound("experiment"))
		return
	}
	if err != nil {
		util.WriteErr(w, err)
		return
	}
	util.WriteJSON(w, 200, e)
}

func (s *Server) handleExperimentDelete(w http.ResponseWriter, r *http.Request) {
	defer s.Store.Lock()()
	res, err := s.Store.DB().ExecContext(r.Context(), `DELETE FROM experiments WHERE id=?`, r.PathValue("id"))
	if err != nil {
		util.WriteErr(w, err)
		return
	}
	if n, _ := res.RowsAffected(); n == 0 {
		util.WriteErr(w, util.ErrNotFound("experiment"))
		return
	}
	util.WriteJSON(w, 200, map[string]any{"deleted": true})
}

func scanExperiment(sc interface{ Scan(...any) error }) (map[string]any, error) {
	var id, jobID, name, taskType, backendID, arch, cfg, metrics, artifacts, runtime, notes, created string
	var seed int64
	var datasetID, datasetVersionID, curves sql.NullString
	if err := sc.Scan(&id, &jobID, &name, &taskType, &backendID, &arch, &seed, &cfg,
		&datasetID, &datasetVersionID, &metrics, &curves, &artifacts, &runtime, &notes, &created); err != nil {
		return nil, err
	}
	e := map[string]any{"id": id, "jobId": jobID, "name": name, "taskType": taskType,
		"backendId": backendID, "architecture": arch, "seed": seed, "notes": notes, "createdAt": created}
	var cfgV, metV, artV, runV any
	_ = json.Unmarshal([]byte(cfg), &cfgV)
	_ = json.Unmarshal([]byte(metrics), &metV)
	_ = json.Unmarshal([]byte(artifacts), &artV)
	_ = json.Unmarshal([]byte(runtime), &runV)
	e["config"] = cfgV
	e["metrics"] = metV
	e["artifacts"] = artV
	e["runtime"] = runV
	if curves.Valid {
		var c any
		_ = json.Unmarshal([]byte(curves.String), &c)
		e["curves"] = c
	}
	e["datasetId"] = datasetID.String
	e["datasetVersionId"] = datasetVersionID.String
	return e, nil
}

// ---- models ----

func (s *Server) handleModelList(w http.ResponseWriter, r *http.Request) {
	rows, err := s.Store.DB().QueryContext(r.Context(),
		`SELECT id, name, task_type, architecture, notes, created_at, updated_at FROM models
                 WHERE project_id=? ORDER BY created_at DESC`, r.PathValue("id"))
	if err != nil {
		util.WriteErr(w, err)
		return
	}
	defer rows.Close()
	out := []map[string]any{}
	type modelRow struct{ id, name, taskType, arch, notes, created, updated string }
	var mrows []modelRow
	for rows.Next() {
		var m modelRow
		if err := rows.Scan(&m.id, &m.name, &m.taskType, &m.arch, &m.notes, &m.created, &m.updated); err != nil {
			util.WriteErr(w, err)
			return
		}
		mrows = append(mrows, m)
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		util.WriteErr(w, err)
		return
	}
	// Version lookup happens after the outer rows are drained — the store
	// allows a single SQLite connection, so nested queries would deadlock.
	for _, m := range mrows {
		versions, err := s.modelVersions(r, m.id)
		if err != nil {
			util.WriteErr(w, err)
			return
		}
		if versions == nil {
			versions = []map[string]any{}
		}
		out = append(out, map[string]any{"id": m.id, "name": m.name, "taskType": m.taskType,
			"architecture": m.arch, "notes": m.notes, "createdAt": m.created,
			"updatedAt": m.updated, "versions": versions})
	}
	util.WriteJSON(w, 200, out)
}

func (s *Server) handleModelGet(w http.ResponseWriter, r *http.Request) {
	mo, err := s.getModel(r, r.PathValue("id"))
	if err != nil {
		util.WriteErr(w, err)
		return
	}
	util.WriteJSON(w, 200, mo)
}

func (s *Server) getModel(r *http.Request, id string) (map[string]any, error) {
	var name, taskType, arch, notes, created, updated, projectID string
	err := s.Store.DB().QueryRowContext(r.Context(),
		`SELECT project_id, name, task_type, architecture, notes, created_at, updated_at FROM models WHERE id=?`, id).
		Scan(&projectID, &name, &taskType, &arch, &notes, &created, &updated)
	if err == sql.ErrNoRows {
		return nil, util.ErrNotFound("model")
	}
	if err != nil {
		return nil, err
	}
	versions, err := s.modelVersions(r, id)
	if err != nil {
		return nil, err
	}
	if versions == nil {
		versions = []map[string]any{}
	}
	return map[string]any{"id": id, "projectId": projectID, "name": name, "taskType": taskType,
		"architecture": arch, "notes": notes, "createdAt": created, "updatedAt": updated, "versions": versions}, nil
}

func (s *Server) handleModelUpdate(w http.ResponseWriter, r *http.Request) {
	var in struct {
		Name  *string `json:"name"`
		Notes *string `json:"notes"`
	}
	if err := util.DecodeJSON(r, &in); err != nil {
		util.WriteErr(w, err)
		return
	}
	defer s.Store.Lock()()
	if in.Name != nil && *in.Name != "" {
		if _, err := s.Store.DB().ExecContext(r.Context(),
			`UPDATE models SET name=?, updated_at=? WHERE id=?`, *in.Name,
			time.Now().UTC().Format(time.RFC3339), r.PathValue("id")); err != nil {
			util.WriteErr(w, err)
			return
		}
	}
	if in.Notes != nil {
		if _, err := s.Store.DB().ExecContext(r.Context(),
			`UPDATE models SET notes=?, updated_at=? WHERE id=?`, *in.Notes,
			time.Now().UTC().Format(time.RFC3339), r.PathValue("id")); err != nil {
			util.WriteErr(w, err)
			return
		}
	}
	util.WriteJSON(w, 200, map[string]any{"updated": true})
}

func (s *Server) handleModelDelete(w http.ResponseWriter, r *http.Request) {
	if r.URL.Query().Get("confirm") != "true" {
		util.WriteErr(w, util.Err(400, "confirm_required", "destructive operation: add ?confirm=true"))
		return
	}
	id := r.PathValue("id")
	// Remove version files, then rows (cascade deletes versions).
	rows, err := s.Store.DB().QueryContext(r.Context(), `SELECT file_path FROM model_versions WHERE model_id=?`, id)
	if err != nil {
		util.WriteErr(w, err)
		return
	}
	var paths []string
	for rows.Next() {
		var p string
		_ = rows.Scan(&p)
		paths = append(paths, p)
	}
	rows.Close()
	defer s.Store.Lock()()
	if _, err := s.Store.DB().ExecContext(r.Context(), `DELETE FROM models WHERE id=?`, id); err != nil {
		util.WriteErr(w, err)
		return
	}
	deleted := 0
	for _, p := range paths {
		if filepath.IsAbs(p) && filepath.Base(p) != "" {
			if os.Remove(p) == nil {
				deleted++
			}
		}
	}
	util.WriteJSON(w, 200, map[string]any{"deleted": true, "filesRemoved": deleted})
}

func (s *Server) modelVersions(r *http.Request, modelID string) ([]map[string]any, error) {
	rows, err := s.Store.DB().QueryContext(r.Context(), `SELECT id, version, COALESCE(experiment_id,''), format,
          file_size, sha256, input_spec, output_spec, classes, metrics, export_formats, license, runtime_reqs, notes, created_at
          FROM model_versions WHERE model_id=? ORDER BY version DESC`, modelID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []map[string]any{}
	for rows.Next() {
		var id, expID, format, sha, inputSpec, outputSpec, classes, metrics, exportFormats, license, runtimeReqs, notes, created string
		var version int
		var size int64
		if err := rows.Scan(&id, &version, &expID, &format, &size, &sha, &inputSpec, &outputSpec,
			&classes, &metrics, &exportFormats, &license, &runtimeReqs, &notes, &created); err != nil {
			return nil, err
		}
		v := map[string]any{"id": id, "modelId": modelID, "version": version, "format": format,
			"fileSize": size, "sha256": sha, "license": license, "notes": notes, "createdAt": created}
		if expID != "" {
			v["experimentId"] = expID
		}
		var inputSpecV any
		_ = json.Unmarshal([]byte(inputSpec), &inputSpecV)
		v["inputSpec"] = inputSpecV
		var outputSpecV any
		_ = json.Unmarshal([]byte(outputSpec), &outputSpecV)
		v["outputSpec"] = outputSpecV
		var classesV any
		_ = json.Unmarshal([]byte(classes), &classesV)
		v["classes"] = classesV
		var metricsV any
		_ = json.Unmarshal([]byte(metrics), &metricsV)
		v["metrics"] = metricsV
		var exportFormatsV any
		_ = json.Unmarshal([]byte(exportFormats), &exportFormatsV)
		v["exportFormats"] = exportFormatsV
		var runtimeReqsV any
		_ = json.Unmarshal([]byte(runtimeReqs), &runtimeReqsV)
		v["runtimeReqs"] = runtimeReqsV
		out = append(out, v)
	}
	return out, rows.Err()
}

func (s *Server) handleModelVersionList(w http.ResponseWriter, r *http.Request) {
	versions, err := s.modelVersions(r, r.PathValue("id"))
	if err != nil {
		util.WriteErr(w, err)
		return
	}
	if versions == nil {
		versions = []map[string]any{}
	}
	util.WriteJSON(w, 200, versions)
}

func (s *Server) handleModelVersionGet(w http.ResponseWriter, r *http.Request) {
	versions, err := s.modelVersions(r, r.PathValue("id"))
	if err != nil {
		util.WriteErr(w, err)
		return
	}
	for _, v := range versions {
		if v["id"] == r.PathValue("vid") {
			util.WriteJSON(w, 200, v)
			return
		}
	}
	util.WriteErr(w, util.ErrNotFound("model version"))
}

// handleModelDownload streams the version's model file with a stable name.
func (s *Server) handleModelDownload(w http.ResponseWriter, r *http.Request) {
	var path, format string
	var version int
	err := s.Store.DB().QueryRowContext(r.Context(),
		`SELECT file_path, format, version FROM model_versions WHERE id=? AND model_id=?`,
		r.PathValue("vid"), r.PathValue("id")).Scan(&path, &format, &version)
	if err == sql.ErrNoRows {
		util.WriteErr(w, util.ErrNotFound("model version"))
		return
	}
	if err != nil {
		util.WriteErr(w, err)
		return
	}
	data, err := os.ReadFile(path)
	if err != nil {
		util.WriteErr(w, util.Err(410, "artifact_missing", "model file is no longer on disk"))
		return
	}
	ext := ".model.json"
	if format != "" {
		ext = "." + format + ".json"
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Disposition",
		fmt.Sprintf("attachment; filename=\"model-v%d%s\"", version, ext))
	w.Header().Set("X-Content-Type-Options", "nosniff")
	_, _ = w.Write(data)
}
