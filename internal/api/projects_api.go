package api

import (
	"io"
	"net/http"
	"os"
	"strconv"

	"neurostudio/internal/domain"
	"neurostudio/internal/projects"
	"neurostudio/internal/util"
)

func (s *Server) handleProjectList(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	list, err := s.Projects.List(r.Context(), q.Get("query"), q.Get("status"), atoiOr(q.Get("limit"), 100))
	if err != nil {
		util.WriteErr(w, err)
		return
	}
	if list == nil {
		list = []*domain.Project{}
	}
	util.WriteJSON(w, 200, list)
}

func (s *Server) handleProjectCreate(w http.ResponseWriter, r *http.Request) {
	var in projects.CreateInput
	if err := util.DecodeJSON(r, &in); err != nil {
		util.WriteErr(w, err)
		return
	}
	p, err := s.Projects.Create(r.Context(), in)
	if err != nil {
		util.WriteErr(w, err)
		return
	}
	util.WriteJSON(w, 201, p)
}

func (s *Server) handleProjectGet(w http.ResponseWriter, r *http.Request) {
	p, err := s.Projects.Get(r.Context(), r.PathValue("id"))
	if err != nil {
		util.WriteErr(w, err)
		return
	}
	counts, _ := s.Projects.Counts(r.Context(), p.ID)
	p.Counts = counts
	util.WriteJSON(w, 200, p)
}

func (s *Server) handleProjectUpdate(w http.ResponseWriter, r *http.Request) {
	var patch map[string]any
	if err := util.DecodeJSON(r, &patch); err != nil {
		util.WriteErr(w, err)
		return
	}
	p, err := s.Projects.Update(r.Context(), r.PathValue("id"), patch)
	if err != nil {
		util.WriteErr(w, err)
		return
	}
	util.WriteJSON(w, 200, p)
}

func (s *Server) handleProjectDelete(w http.ResponseWriter, r *http.Request) {
	if r.URL.Query().Get("confirm") != "true" {
		util.WriteErr(w, util.Err(400, "confirm_required",
			"destructive operation: add ?confirm=true (the UI shows a confirmation dialog)"))
		return
	}
	if err := s.Projects.Delete(r.Context(), r.PathValue("id")); err != nil {
		util.WriteErr(w, err)
		return
	}
	util.WriteJSON(w, 200, map[string]any{"deleted": true})
}

func (s *Server) handleProjectDuplicate(w http.ResponseWriter, r *http.Request) {
	p, err := s.Projects.Duplicate(r.Context(), r.PathValue("id"))
	if err != nil {
		util.WriteErr(w, err)
		return
	}
	util.WriteJSON(w, 201, p)
}

func (s *Server) handleProjectArchive(w http.ResponseWriter, r *http.Request) {
	p, err := s.Projects.SetStatus(r.Context(), r.PathValue("id"), "archived")
	if err != nil {
		util.WriteErr(w, err)
		return
	}
	util.WriteJSON(w, 200, p)
}

func (s *Server) handleProjectUnarchive(w http.ResponseWriter, r *http.Request) {
	p, err := s.Projects.SetStatus(r.Context(), r.PathValue("id"), "active")
	if err != nil {
		util.WriteErr(w, err)
		return
	}
	util.WriteJSON(w, 200, p)
}

// handleProjectExport streams a portable project ZIP (manifest + dataset
// metadata). Binary sample files are exported per-dataset with their own
// endpoints.
func (s *Server) handleProjectExport(w http.ResponseWriter, r *http.Request) {
	path, size, err := s.Projects.ExportProject(r.Context(), r.PathValue("id"))
	if err != nil {
		util.WriteErr(w, err)
		return
	}
	defer os.Remove(path)
	f, err := os.Open(path)
	if err != nil {
		util.WriteErr(w, err)
		return
	}
	defer f.Close()
	w.Header().Set("Content-Type", "application/zip")
	w.Header().Set("Content-Disposition",
		"attachment; filename=\"neurostudio-project-"+r.PathValue("id")+".zip\"")
	w.Header().Set("Content-Length", strconv.FormatInt(size, 10))
	_, _ = io.Copy(w, f)
}

func atoiOr(s string, def int) int {
	if n, err := strconv.Atoi(s); err == nil && n > 0 {
		return n
	}
	return def
}
