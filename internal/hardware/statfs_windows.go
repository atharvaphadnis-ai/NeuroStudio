//go:build windows

package hardware

import (
	"golang.org/x/sys/windows"
)

// DiskFreeGB reports free space on the volume containing path (windows).
func DiskFreeGB(path string) float64 {
	var free, total, avail uint64
	p, err := windows.UTF16PtrFromString(path)
	if err != nil {
		return -1
	}
	if err := windows.GetDiskFreeSpaceEx(p, &free, &total, &avail); err != nil {
		return -1
	}
	return float64(free) / 1e9
}
