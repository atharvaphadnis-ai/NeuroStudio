// NeuroStudio — a self-hosted AI training and development platform.
//
// The binary starts a local HTTP server, serves the embedded frontend,
// initializes SQLite storage, discovers training backends and exposes the
// versioned REST API under /api/v1.
package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"neurostudio/internal/annotations"
	"neurostudio/internal/api"
	"neurostudio/internal/app"
	"neurostudio/internal/backends"
	"neurostudio/internal/backends/goruntime"
	"neurostudio/internal/datasets"
	"neurostudio/internal/projects"
	"neurostudio/internal/store"
	"neurostudio/internal/training"
	"neurostudio/internal/util"
	"neurostudio/internal/webfs"
)

func main() {
	os.Exit(run())
}

func run() int {
	var (
		host        = flag.String("host", "127.0.0.1", "bind address (default localhost only)")
		port        = flag.Int("port", 8765, "HTTP port")
		dataDir     = flag.String("data-dir", "", "data directory (default: OS user-data location or $NEUROSTUDIO_DATA)")
		devFrontend = flag.String("dev-frontend", "", "serve frontend from this directory instead of embedded assets (development mode)")
		allowRemote = flag.Bool("allow-remote", false, "bind non-loopback host; requires deliberate opt-in and enables token auth")
		verbose     = flag.Bool("verbose", false, "verbose logging")
		showVersion = flag.Bool("version", false, "print version and exit")
	)
	flag.Parse()

	if *showVersion {
		fmt.Printf("%s %s\n", app.AppName, app.Version)
		return 0
	}

	level := slog.LevelInfo
	if *verbose {
		level = slog.LevelDebug
	}
	log := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: level}))

	cfg, err := app.ResolveConfig(*host, *port, *dataDir, *devFrontend, *allowRemote, *verbose)
	if err != nil {
		fmt.Fprintln(os.Stderr, "configuration error:", err)
		return 1
	}
	a := &app.App{Config: cfg, Started: time.Now()}
	if err := a.InitPaths(); err != nil {
		fmt.Fprintln(os.Stderr, "storage error:", err)
		return 1
	}

	// Apply a staged restore (if any) BEFORE opening the database.
	if err := applyPendingRestore(a.Paths); err != nil {
		log.Error("pending restore failed", "err", err)
	}

	st, err := store.Open(a.Paths.Database)
	if err != nil {
		fmt.Fprintln(os.Stderr, "database error:", err)
		return 1
	}
	defer st.Close()

	// Recover from interrupted runs: any job still marked queued/running was
	// interrupted by a process exit. We never silently re-run expensive
	// training; jobs are marked failed with a clear reason and their
	// checkpoints remain available.
	if n, err := recoverInterruptedJobs(st); err != nil {
		log.Warn("job recovery failed", "err", err)
	} else if n > 0 {
		log.Warn("recovered interrupted jobs", "marked_failed", n)
	}

	pj := projects.NewService(st)
	ds := datasets.NewService(st, a.Paths.Datasets)
	an := annotations.NewService(st)

	// Training backends: the built-in Go engine is always present.
	// External workers (PyTorch/Keras) register here in later stages.
	reg := backends.NewRegistry(goruntime.New(log))
	tr := training.NewManager(st, reg, a.Paths, log)
	tr.Start()
	defer tr.Close()

	srv := api.NewServer(a, st, pj, ds, an, tr, log)

	assets := webfs.Assets()
	if cfg.DevFrontend != "" {
		log.Info("development mode: serving frontend from disk", "dir", cfg.DevFrontend)
		assets = os.DirFS(cfg.DevFrontend)
	}
	static := srv.StaticHandler(assets)
	mux := http.NewServeMux()
	mux.Handle("/api/", srv.Handler())
	mux.Handle("/", static)

	addr := net.JoinHostPort(cfg.Host, fmt.Sprint(cfg.Port))
	httpSrv := &http.Server{
		Addr:              addr,
		Handler:           mux,
		ReadHeaderTimeout: 15 * time.Second,
		IdleTimeout:       120 * time.Second,
	}

	ln, err := net.Listen("tcp", addr)
	if err != nil {
		fmt.Fprintln(os.Stderr, "cannot listen on", addr, ":", err)
		return 1
	}

	fmt.Printf(`
  %s %s
  Local URL:  http://%s
  Data dir:   %s
  Telemetry:  disabled
`, app.AppName, app.Version, addr, cfg.DataDir)
	if cfg.AllowRemote {
		fmt.Printf("  API token:  %s  (required for non-local requests)\n", cfg.AuthToken)
	}
	for _, w := range a.WarnHealth() {
		fmt.Println("  warning:", w)
	}
	fmt.Println("  Press Ctrl+C to stop.")

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	errCh := make(chan error, 1)
	go func() { errCh <- httpSrv.Serve(ln) }()

	select {
	case <-ctx.Done():
		log.Info("shutting down…")
		shCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_ = httpSrv.Shutdown(shCtx)
		return 0
	case err := <-errCh:
		if err != nil && err != http.ErrServerClosed {
			fmt.Fprintln(os.Stderr, "server error:", err)
			return 1
		}
		return 0
	}
}

// applyPendingRestore swaps in a staged backup before the store opens.
func applyPendingRestore(p app.Paths) error {
	marker := filepath.Join(p.Root, "pending-restore")
	data, err := os.ReadFile(marker)
	if err != nil {
		return nil // nothing staged
	}
	src := filepath.Join(p.Backups, filepath.Base(string(data)))
	if err := store.Restore(p.Database, src); err != nil {
		os.Remove(marker)
		return err
	}
	os.Remove(marker)
	return nil
}

// recoverInterruptedJobs marks jobs that were running when the process died.
func recoverInterruptedJobs(st *store.Store) (int64, error) {
	res, err := st.DB().Exec(`UPDATE jobs
          SET status='failed',
              error='application exited while this job was running; checkpoints were preserved and can seed a new run',
              finished_at=?
          WHERE status IN ('queued','running')`,
		time.Now().UTC().Format(time.RFC3339))
	if err != nil {
		return 0, err
	}
	return res.RowsAffected()
}

var _ = util.NewID
