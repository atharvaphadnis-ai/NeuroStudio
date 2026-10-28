//go:build !windows

package hardware

import "syscall"

// DiskFreeGB reports free space on the volume containing path (unix).
func DiskFreeGB(path string) float64 {
	var st syscall.Statfs_t
	if err := syscall.Statfs(path, &st); err != nil {
		return -1
	}
	return float64(st.Bavail*uint64(st.Bsize)) / 1e9
}
