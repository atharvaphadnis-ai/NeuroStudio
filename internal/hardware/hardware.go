// Package hardware reports the actual capabilities of the machine: CPU,
// memory, disk and optional accelerators. Detection is best-effort and
// honest — when a value cannot be determined it is reported as unknown
// rather than guessed.
package hardware

import (
	"bufio"
	"context"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"time"
)

type Info struct {
	OS         string  `json:"os"`
	Arch       string  `json:"arch"`
	GoVersion  string  `json:"goVersion"`
	CPUModel   string  `json:"cpuModel"`
	CPUCores   int     `json:"cpuCores"`
	RAMTotalMB int64   `json:"ramTotalMb"`
	RAMFreeMB  int64   `json:"ramFreeMb"`
	DiskFreeGB float64 `json:"diskFreeGb"`
	GPUs       []GPU   `json:"gpus"`
}

type GPU struct {
	Name        string `json:"name"`
	VRAMMB      int64  `json:"vramMb"`
	Driver      string `json:"driver,omitempty"`
	ComputeCap  string `json:"computeCapability,omitempty"`
	CUDAVersion string `json:"cudaVersion,omitempty"`
	Note        string `json:"note,omitempty"`
}

func Detect(dataDir string) Info {
	info := Info{
		OS:        runtime.GOOS,
		Arch:      runtime.GOARCH,
		GoVersion: runtime.Version(),
		CPUCores:  runtime.NumCPU(),
		GPUs:      []GPU{},
	}
	info.CPUModel = cpuModel()
	info.RAMTotalMB, info.RAMFreeMB = memory()
	info.DiskFreeGB = DiskFreeGB(dataDir)
	if g := detectNvidia(); g != nil {
		info.GPUs = append(info.GPUs, *g)
	}
	return info
}

func cpuModel() string {
	if runtime.GOOS != "linux" {
		return ""
	}
	f, err := os.Open("/proc/cpuinfo")
	if err != nil {
		return ""
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := sc.Text()
		if strings.HasPrefix(line, "model name") {
			if i := strings.Index(line, ":"); i >= 0 {
				return strings.TrimSpace(line[i+1:])
			}
		}
	}
	return ""
}

func memory() (total, freeMB int64) {
	if runtime.GOOS != "linux" {
		return -1, -1
	}
	f, err := os.Open("/proc/meminfo")
	if err != nil {
		return -1, -1
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		fields := strings.Fields(sc.Text())
		if len(fields) < 2 {
			continue
		}
		v, _ := strconv.ParseInt(fields[1], 10, 64)
		switch fields[0] {
		case "MemTotal:":
			total = v / 1024
		case "MemAvailable:":
			freeMB = v / 1024
		}
	}
	return total, freeMB
}

// detectNvidia probes for an NVIDIA GPU via nvidia-smi. Absence of the tool
// is not an error — most machines have no NVIDIA card.
func detectNvidia() *GPU {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "nvidia-smi",
		"--query-gpu=name,memory.total,driver_version",
		"--format=csv,noheader,nounits").Output()
	if err != nil {
		return nil
	}
	line := strings.TrimSpace(strings.SplitN(string(out), "\n", 2)[0])
	parts := strings.Split(line, ",")
	if len(parts) < 3 {
		return nil
	}
	vram, _ := strconv.ParseInt(strings.TrimSpace(parts[1]), 10, 64)
	return &GPU{
		Name:   strings.TrimSpace(parts[0]),
		VRAMMB: vram,
		Driver: strings.TrimSpace(parts[2]),
		Note:   "detected via nvidia-smi; usable only by backends with CUDA support",
	}
}
