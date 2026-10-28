package api

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"time"

	"neurostudio/internal/app"
	"neurostudio/internal/hardware"
	"neurostudio/internal/util"
)

func (s *Server) handleSystemInfo(w http.ResponseWriter, r *http.Request) {
	util.WriteJSON(w, 200, map[string]any{
		"name":        app.AppName,
		"version":     app.Version,
		"goVersion":   runtime.Version(),
		"os":          runtime.GOOS,
		"arch":        runtime.GOARCH,
		"dataDir":     s.App.Config.DataDir,
		"startedAt":   s.App.Started.Format(time.RFC3339),
		"uptimeSec":   int(time.Since(s.App.Started).Seconds()),
		"allowRemote": s.App.Config.AllowRemote,
		"telemetry":   "disabled (NeuroStudio collects no telemetry)",
		"apiVersion":  "v1",
	})
}

func (s *Server) handleDiagnostics(w http.ResponseWriter, r *http.Request) {
	hw := hardware.Detect(s.App.Config.DataDir)
	checks := []map[string]any{}
	addCheck := func(name, status, detail string) {
		checks = append(checks, map[string]any{"name": name, "status": status, "detail": detail})
	}
	addCheck("data directory", "ok", s.App.Config.DataDir)
	if res, err := s.Store.IntegrityCheck(r.Context()); err != nil {
		addCheck("database integrity", "fail", err.Error())
	} else {
		addCheck("database integrity", "ok", res)
	}
	if hw.RAMTotalMB < 0 {
		addCheck("memory", "unknown", "platform detection unavailable")
	} else {
		addCheck("memory", "ok", fmt.Sprintf("%d MB total, %d MB available", hw.RAMTotalMB, hw.RAMFreeMB))
	}
	if len(hw.GPUs) == 0 {
		addCheck("gpu", "none", "no NVIDIA GPU detected; CPU training available for supported tasks")
	}
	for _, w := range s.App.WarnHealth() {
		addCheck("disk", "warn", w)
	}
	util.WriteJSON(w, 200, map[string]any{"hardware": hw, "checks": checks})
}

func (s *Server) handleStorage(w http.ResponseWriter, r *http.Request) {
	dbSize := int64(0)
	if st, err := os.Stat(s.Store.Path()); err == nil {
		dbSize = st.Size()
	}
	util.WriteJSON(w, 200, map[string]any{
		"database":   dbSize,
		"datasets":   util.DirSize(s.App.Paths.Datasets),
		"models":     util.DirSize(s.App.Paths.Models),
		"jobs":       util.DirSize(s.App.Paths.Jobs),
		"exports":    util.DirSize(s.App.Paths.Exports),
		"diskFreeGb": hardware.Detect(s.App.Config.DataDir).DiskFreeGB,
	})
}

// handleBackup writes a consistent DB snapshot to the backups directory and
// returns its filename. Binary artifacts stay on disk; backup + the data dir
// together form a complete restore point (documented).
func (s *Server) handleBackup(w http.ResponseWriter, r *http.Request) {
	name := "neurostudio-backup-" + time.Now().UTC().Format("20060102-150405") + ".db"
	dest := filepath.Join(s.App.Paths.Backups, name)
	ctx, cancel := context.WithTimeout(r.Context(), 60*time.Second)
	defer cancel()
	if err := s.Store.Backup(ctx, dest); err != nil {
		util.WriteErr(w, util.Err(500, "backup_failed", err.Error()))
		return
	}
	st, _ := os.Stat(dest)
	util.WriteJSON(w, 200, map[string]any{"file": name, "size": st.Size()})
}

func (s *Server) handleRestore(w http.ResponseWriter, r *http.Request) {
	var req struct {
		File string `json:"file"`
	}
	if err := util.DecodeJSON(r, &req); err != nil {
		util.WriteErr(w, err)
		return
	}
	var running int
	_ = s.Store.DB().QueryRowContext(r.Context(),
		`SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')`).Scan(&running)
	if running > 0 {
		util.WriteErr(w, util.Err(409, "jobs_running",
			"cannot restore while training jobs are active"))
		return
	}
	// Stage the restore: the swap happens on next startup before the DB is
	// opened, which is the only safe moment to replace the live database.
	src := filepath.Base(req.File) // no traversal
	srcPath := filepath.Join(s.App.Paths.Backups, src)
	if _, err := os.Stat(srcPath); err != nil {
		util.WriteErr(w, util.ErrNotFound("backup file"))
		return
	}
	if err := os.WriteFile(filepath.Join(s.App.Paths.Root, "pending-restore"), []byte(src), 0o644); err != nil {
		util.WriteErr(w, util.Err(500, "restore_failed", err.Error()))
		return
	}
	util.WriteJSON(w, 200, map[string]any{
		"staged": src,
		"note":   "restore will be applied on next application start",
	})
}
