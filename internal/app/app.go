// Package app wires configuration, data-directory layout and shared
// services together. NeuroStudio is local-first: one process, one data
// directory, no cloud dependencies.
package app

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"time"

	"neurostudio/internal/hardware"
	"neurostudio/internal/util"
)

const AppName = "NeuroStudio"
const Version = "0.1.0"

// Config is the resolved application configuration.
type Config struct {
	Host        string
	Port        int
	DataDir     string
	DevFrontend string // serve frontend from disk instead of embedded assets
	AllowRemote bool
	AuthToken   string // only used when AllowRemote is true
	MaxUploadMB int64
	Verbose     bool
}

// Paths describes the layout under the data directory.
type Paths struct {
	Root       string
	Database   string
	Datasets   string // dataset sample files
	Models     string // trained model artifacts
	Jobs       string // per-job checkpoints and logs
	Exports    string // generated export bundles
	Backups    string
	UploadsTmp string
}

type App struct {
	Config  Config
	Paths   Paths
	Started time.Time
}

// ResolveConfig applies defaults and validates flags.
func ResolveConfig(host string, port int, dataDir, devFrontend string, allowRemote bool, verbose bool) (Config, error) {
	if dataDir == "" {
		dataDir = DefaultDataDir()
	}
	if err := os.MkdirAll(dataDir, 0o755); err != nil {
		return Config{}, fmt.Errorf("cannot create data dir %s: %w", dataDir, err)
	}
	abs, err := filepath.Abs(dataDir)
	if err != nil {
		return Config{}, err
	}
	cfg := Config{
		Host:        host,
		Port:        port,
		DataDir:     abs,
		DevFrontend: devFrontend,
		AllowRemote: allowRemote,
		MaxUploadMB: 300,
		Verbose:     verbose,
	}
	if port < 1 || port > 65535 {
		return Config{}, fmt.Errorf("invalid port %d", port)
	}
	if !allowRemote && host != "127.0.0.1" && host != "localhost" && host != "" {
		return Config{}, errors.New("binding to a non-loopback host requires --allow-remote (see docs/SECURITY details in README)")
	}
	if allowRemote {
		cfg.Host = host
		cfg.AuthToken = util.NewID() + util.NewID()
	}
	return cfg, nil
}

func DefaultDataDir() string {
	if v := os.Getenv("NEUROSTUDIO_DATA"); v != "" {
		return v
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "./neurostudio-data"
	}
	switch runtime.GOOS {
	case "windows":
		return filepath.Join(home, "AppData", "Local", "NeuroStudio")
	case "darwin":
		return filepath.Join(home, "Library", "Application Support", "NeuroStudio")
	default:
		return filepath.Join(home, ".local", "share", "neurostudio")
	}
}

// InitPaths creates the data-directory layout.
func (a *App) InitPaths() error {
	p := Paths{
		Root:       a.Config.DataDir,
		Database:   filepath.Join(a.Config.DataDir, "neurostudio.db"),
		Datasets:   filepath.Join(a.Config.DataDir, "datasets"),
		Models:     filepath.Join(a.Config.DataDir, "models"),
		Jobs:       filepath.Join(a.Config.DataDir, "jobs"),
		Exports:    filepath.Join(a.Config.DataDir, "exports"),
		Backups:    filepath.Join(a.Config.DataDir, "backups"),
		UploadsTmp: filepath.Join(a.Config.DataDir, "tmp-uploads"),
	}
	for _, d := range []string{p.Root, p.Datasets, p.Models, p.Jobs, p.Exports, p.Backups, p.UploadsTmp} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			return err
		}
	}
	a.Paths = p
	return nil
}

// WarnHealth reports data-dir problems at startup (disk space, permissions).
func (a *App) WarnHealth() []string {
	var warns []string
	if free := hardware.DiskFreeGB(a.Config.DataDir); free >= 0 && free < 1.0 {
		warns = append(warns, fmt.Sprintf("less than 1 GB free on data volume (%.2f GB)", free))
	}
	return warns
}
