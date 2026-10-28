// Package api implements the versioned HTTP API (/api/v1), static frontend
// serving, SSE streaming and middleware. Routes use Go 1.22+ ServeMux method
// patterns; identifiers come from r.PathValue.
package api

import (
	"io/fs"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"neurostudio/internal/annotations"
	"neurostudio/internal/app"
	"neurostudio/internal/datasets"
	"neurostudio/internal/projects"
	"neurostudio/internal/store"
	"neurostudio/internal/training"
	"neurostudio/internal/util"
)

type Server struct {
	App         *app.App
	Store       *store.Store
	Projects    *projects.Service
	Datasets    *datasets.Service
	Annotations *annotations.Service
	Trainer     *training.Manager
	Log         *slog.Logger
	mux         *http.ServeMux
}

func NewServer(a *app.App, st *store.Store, pj *projects.Service, ds *datasets.Service, an *annotations.Service, tr *training.Manager, log *slog.Logger) *Server {
	s := &Server{App: a, Store: st, Projects: pj, Datasets: ds, Annotations: an, Trainer: tr, Log: log, mux: http.NewServeMux()}
	s.routes()
	return s
}

func (s *Server) routes() {
	m := s.mux
	// System.
	m.HandleFunc("GET /api/v1/system/info", s.handleSystemInfo)
	m.HandleFunc("GET /api/v1/system/diagnostics", s.handleDiagnostics)
	m.HandleFunc("GET /api/v1/system/storage", s.handleStorage)
	m.HandleFunc("POST /api/v1/system/backup", s.handleBackup)
	m.HandleFunc("POST /api/v1/system/restore", s.handleRestore)

	// Projects.
	m.HandleFunc("GET /api/v1/projects", s.handleProjectList)
	m.HandleFunc("POST /api/v1/projects", s.handleProjectCreate)
	m.HandleFunc("GET /api/v1/projects/{id}", s.handleProjectGet)
	m.HandleFunc("PATCH /api/v1/projects/{id}", s.handleProjectUpdate)
	m.HandleFunc("DELETE /api/v1/projects/{id}", s.handleProjectDelete)
	m.HandleFunc("POST /api/v1/projects/{id}/duplicate", s.handleProjectDuplicate)
	m.HandleFunc("POST /api/v1/projects/{id}/archive", s.handleProjectArchive)
	m.HandleFunc("POST /api/v1/projects/{id}/restore", s.handleProjectUnarchive)
	m.HandleFunc("GET /api/v1/projects/{id}/export", s.handleProjectExport)

	// Registered by feature packages during later stages:
	s.registerDatasetRoutes(m)
	s.registerAnnotationRoutes(m)
	s.registerTrainingRoutes(m)
	s.registerRegistryRoutes(m)
	s.registerInferenceRoutes(m)
}

func (s *Server) Handler() http.Handler {
	var h http.Handler = s.mux
	h = s.recoverMiddleware(h)
	if s.App.Config.AllowRemote {
		h = s.authMiddleware(h)
	}
	h = s.logMiddleware(h)
	return s.securityHeaders(h)
}

// ---- middleware ----

func (s *Server) recoverMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if rec := recover(); rec != nil {
				s.Log.Error("panic recovered", "path", r.URL.Path, "err", rec)
				util.WriteErr(w, util.Err(500, "internal", "internal server error"))
			}
		}()
		next.ServeHTTP(w, r)
	})
}

func (s *Server) logMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		if strings.HasPrefix(r.URL.Path, "/api/") {
			s.Log.Debug("request", "method", r.Method, "path", r.URL.Path,
				"duration", time.Since(start).Round(time.Millisecond).String())
		}
	})
}

func (s *Server) authMiddleware(next http.Handler) http.Handler {
	token := s.App.Config.AuthToken
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Static SPA shell is served so the user can enter the token in the
		// UI; everything else requires the bearer token.
		if r.URL.Path != "/" && !strings.HasPrefix(r.URL.Path, "/assets/") {
			auth := r.Header.Get("Authorization")
			// EventSource cannot set headers; SSE may pass ?token=.
			if auth == "" && r.URL.Query().Get("token") == token {
				auth = "Bearer " + token
			}
			if !strings.EqualFold(auth, "Bearer "+token) {
				w.Header().Set("WWW-Authenticate", `Bearer realm="neurostudio"`)
				util.WriteErr(w, util.Err(401, "unauthorized",
					"remote access requires the API token (see server console)"))
				return
			}
		}
		next.ServeHTTP(w, r)
	})
}

func (s *Server) securityHeaders(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		h := w.Header()
		h.Set("X-Content-Type-Options", "nosniff")
		h.Set("X-Frame-Options", "DENY")
		h.Set("Referrer-Policy", "no-referrer")
		next.ServeHTTP(w, r)
	})
}

// ---- static frontend ----

// StaticHandler serves the embedded (or dev-mode) SPA. Unknown paths fall
// back to index.html for client-side routing.
func (s *Server) StaticHandler(assets fs.FS) http.HandlerFunc {
	fileServer := http.FileServerFS(assets)
	return func(w http.ResponseWriter, r *http.Request) {
		path := strings.TrimPrefix(r.URL.Path, "/")
		if path == "" {
			path = "index.html"
		}
		if _, err := fs.Stat(assets, path); err != nil {
			// SPA fallback.
			r.URL.Path = "/"
			data, err2 := fs.ReadFile(assets, "index.html")
			if err2 != nil {
				http.Error(w, "frontend assets missing", http.StatusInternalServerError)
				return
			}
			w.Header().Set("Content-Type", "text/html; charset=utf-8")
			w.Header().Set("Cache-Control", "no-cache")
			_, _ = w.Write(data)
			return
		}
		if strings.HasPrefix(path, "js/") || strings.HasPrefix(path, "css/") {
			w.Header().Set("Cache-Control", "no-cache")
		}
		fileServer.ServeHTTP(w, r)
	}
}
