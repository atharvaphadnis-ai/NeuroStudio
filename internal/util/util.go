// Package util provides small shared helpers: IDs, JSON decoding, safe
// filesystem paths and HTTP response helpers. Filesystem helpers enforce
// strict path-traversal protection — all user-influenced paths must resolve
// inside the data directory.
package util

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

// NewID returns a random 128-bit identifier, hex encoded.
func NewID() string {
	b := make([]byte, 16)
	if _, err := rand.Read(b); err != nil {
		panic("crypto/rand unavailable: " + err.Error())
	}
	return hex.EncodeToString(b)
}

// DecodeJSON decodes a request body into dst with unknown fields rejected,
// keeping API contracts strict.
func DecodeJSON(r *http.Request, dst any) error {
	dec := json.NewDecoder(http.MaxBytesReader(nil, r.Body, 4<<20))
	dec.DisallowUnknownFields()
	if err := dec.Decode(dst); err != nil {
		return fmt.Errorf("invalid JSON body: %w", err)
	}
	return nil
}

// WriteJSON writes a JSON response with the given status code.
func WriteJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(v)
}

// APIError is the standard error envelope.
type APIError struct {
	Status  int    `json:"status"`
	Code    string `json:"code"`
	Message string `json:"message"`
}

func (e *APIError) Error() string { return e.Message }

func Err(status int, code, msg string) *APIError {
	return &APIError{Status: status, Code: code, Message: msg}
}

func ErrNotFound(what string) *APIError {
	return Err(http.StatusNotFound, "not_found", what+" not found")
}

// APIErrCode returns the error code of an *APIError, or "" for other errors.
func APIErrCode(err error) string {
	var ae *APIError
	if errors.As(err, &ae) {
		return ae.Code
	}
	return ""
}

func ErrBadRequest(format string, a ...any) *APIError {
	return Err(http.StatusBadRequest, "bad_request", fmt.Sprintf(format, a...))
}

var ErrConflict = Err(http.StatusConflict, "conflict", "conflicting state")

// WriteErr writes an error envelope; unknown errors become 500 without
// leaking internals.
func WriteErr(w http.ResponseWriter, err error) {
	var ae *APIError
	if errors.As(err, &ae) {
		WriteJSON(w, ae.Status, ae)
		return
	}
	WriteJSON(w, http.StatusInternalServerError, &APIError{
		Status: 500, Code: "internal", Message: "internal server error",
	})
}

// ErrCancelled is returned when a job was cancelled by user request.
var ErrCancelled = errors.New("cancelled by user")

// SafeJoin joins base and rel, refusing absolute paths, any ".." component,
// and ensuring the result stays inside base. This is the single choke point
// for turning untrusted names into filesystem paths.
func SafeJoin(base, rel string) (string, error) {
	if rel == "" {
		return "", errors.New("empty path")
	}
	if strings.HasPrefix(rel, "/") || strings.Contains(rel, "\x00") {
		return "", fmt.Errorf("invalid path %q", rel)
	}
	if strings.Contains(rel, "..") {
		return "", fmt.Errorf("path must not contain '..': %q", rel)
	}
	clean := filepath.Clean("/" + rel) // forces rooted, removes redundancies
	rel2 := strings.TrimPrefix(clean, "/")
	if rel2 == "" || rel2 == "." {
		return "", fmt.Errorf("invalid path %q", rel)
	}
	full := filepath.Join(base, rel2)
	absBase, err := filepath.Abs(base)
	if err != nil {
		return "", err
	}
	absFull, err := filepath.Abs(full)
	if err != nil {
		return "", err
	}
	if absFull != absBase && !strings.HasPrefix(absFull, absBase+string(filepath.Separator)) {
		return "", fmt.Errorf("path escapes data directory: %q", rel)
	}
	return full, nil
}

// AtomicWrite writes data to path via a temp file + rename so partial files
// never appear at the target location.
func AtomicWrite(path string, data []byte, perm fs.FileMode) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, ".tmp-*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := os.Chmod(tmpName, perm); err != nil {
		return err
	}
	return os.Rename(tmpName, path)
}

// DirSize recursively sums file sizes under root (0 if missing).
func DirSize(root string) int64 {
	var total int64
	_ = filepath.WalkDir(root, func(_ string, d fs.DirEntry, err error) error {
		if err == nil && d.Type().IsRegular() {
			if info, err := d.Info(); err == nil {
				total += info.Size()
			}
		}
		return nil
	})
	return total
}

// HumanBytes formats a byte count for display.
func HumanBytes(n int64) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	div, exp := int64(unit), 0
	for m := n / unit; m >= unit; m /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(n)/float64(div), "KMGTPE"[exp])
}
