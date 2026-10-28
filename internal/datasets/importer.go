package datasets

import (
	"archive/zip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"image"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"

	"neurostudio/internal/domain"
	"neurostudio/internal/util"
)

// maxFileBytes per uploaded sample (25 MB covers images and short audio).
const maxFileBytes = 25 << 20

// allowedExtensions limits what can be stored, as defense-in-depth beyond
// content-type sniffing.
var allowedExt = map[string]string{
	".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
	".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
	".wav": "audio/wav", ".mp3": "audio/mpeg", ".ogg": "audio/ogg", ".webm": "audio/webm", ".m4a": "audio/mp4",
	".mp4": "video/mp4", ".mov": "video/quicktime",
	".csv": "text/csv", ".tsv": "text/tab-separated-values", ".json": "application/json",
	".txt": "text/plain", ".md": "text/plain",
}

// ImportResult reports what happened during an import batch.
type ImportResult struct {
	Added      int      `json:"added"`
	Duplicates int      `json:"duplicates"`
	Corrupt    int      `json:"corrupt"`
	Rejected   int      `json:"rejected"`
	Errors     []string `json:"errors,omitempty"`
	Classes    []string `json:"classes,omitempty"`
}

// ImportMultipart imports uploaded files. Files may arrive with a class hint
// (form field "class") and/or structured field names like
// "files.<ClassName>" produced by folder drag-and-drop, which auto-creates
// classes from the directory name.
func (s *Service) ImportMultipart(ctx context.Context, ds *domain.Dataset, r *http.Request, defaultClass string) (*ImportResult, error) {
	res := &ImportResult{}
	mr, err := r.MultipartReader()
	if err != nil {
		return nil, util.ErrBadRequest("expected multipart form: %v", err)
	}
	// map from class name -> class id (lazily created)
	classIDs := map[string]string{}
	ensureClass := func(name string) (string, error) {
		name = strings.TrimSpace(name)
		if name == "" {
			return "", nil
		}
		if id, ok := classIDs[name]; ok {
			return id, nil
		}
		c, err := s.AddClass(ctx, ds.ID, name)
		if err != nil {
			if util.APIErrCode(err) == "duplicate_class" {
				classes, _ := s.ListClasses(ctx, ds.ID)
				for _, cc := range classes {
					if cc.Name == name {
						classIDs[name] = cc.ID
						return cc.ID, nil
					}
				}
			}
			return "", err
		}
		classIDs[name] = c.ID
		return c.ID, nil
	}

	if defaultClass != "" {
		if id, err := ensureClass(defaultClass); err == nil && id != "" {
			// pre-registered
		}
	}

	for {
		part, err := mr.NextPart()
		if err == io.EOF {
			break
		}
		if err != nil {
			return res, err
		}
		if part.FormName() == "class" {
			data, _ := io.ReadAll(io.LimitReader(part, 256))
			if cname := strings.TrimSpace(string(data)); cname != "" {
				ensureClass(cname)
			}
			continue
		}
		if part.FileName() == "" {
			continue
		}
		// Class hint from field name "files.<Class>".
		classHint := defaultClass
		if strings.HasPrefix(part.FormName(), "files.") {
			classHint = strings.TrimPrefix(part.FormName(), "files.")
		}
		classID, ok := classIDs[classHint]
		if classHint != "" && !ok {
			classID, err = ensureClass(classHint)
			if err != nil {
				res.Errors = append(res.Errors, fmt.Sprintf("class %q: %v", classHint, err))
				classID = ""
			}
		}
		added, dup, corrupt, rejected, errMsg := s.storeOne(ctx, ds, part, part.FileName(), classID)
		res.Added += added
		res.Duplicates += dup
		res.Corrupt += corrupt
		res.Rejected += rejected
		if errMsg != "" {
			res.Errors = append(res.Errors, errMsg)
		}
	}
	s.touchDataset(ctx, ds.ID)
	return res, nil
}

func (s *Service) storeOne(ctx context.Context, ds *domain.Dataset, src io.Reader, filename string, classID string) (added, dup, corrupt, rejected int, errMsg string) {
	if err := os.MkdirAll(s.dataRoot, 0o755); err != nil {
		errMsg = err.Error()
		return
	}
	filename = filepath.Base(strings.ReplaceAll(filename, "\\", "/"))
	if filename == "" || filename == "." || strings.HasPrefix(filename, ".") {
		rejected = 1
		return
	}
	ext := strings.ToLower(filepath.Ext(filename))
	mt, ok := allowedExt[ext]
	if !ok {
		rejected = 1
		errMsg = fmt.Sprintf("%s: unsupported file type", filename)
		return
	}
	tmp, err := os.CreateTemp(s.dataRoot, ".upload-*")
	if err != nil {
		errMsg = err.Error()
		return
	}
	defer os.Remove(tmp.Name())
	h := sha256.New()
	n, err := io.Copy(io.MultiWriter(tmp, h), io.LimitReader(src, maxFileBytes+1))
	if err != nil {
		tmp.Close()
		errMsg = fmt.Sprintf("%s: read error", filename)
		return
	}
	tmp.Close()
	if n > maxFileBytes {
		rejected = 1
		errMsg = fmt.Sprintf("%s: file exceeds %d MB limit", filename, maxFileBytes>>20)
		return
	}
	if n == 0 {
		rejected = 1
		errMsg = fmt.Sprintf("%s: empty file", filename)
		return
	}
	sum := hex.EncodeToString(h.Sum(nil))

	// Duplicate detection (same dataset, same content hash).
	var existing string
	_ = s.st.DB().QueryRowContext(ctx,
		`SELECT rel_path FROM samples WHERE dataset_id=? AND sha256=? LIMIT 1`, ds.ID, sum).Scan(&existing)

	// Content validation: images must decode; corrupt files get flagged,
	// stored for inspection, and excluded from splits/training.
	width, height := 0, 0
	if strings.HasPrefix(mt, "image/") {
		f, err := os.Open(tmp.Name())
		if err == nil {
			cfg, _, err := image.DecodeConfig(f)
			f.Close()
			if err != nil {
				corrupt = 1
			} else if cfg.Width > 8000 || cfg.Height > 8000 || cfg.Width == 0 || cfg.Height == 0 {
				corrupt = 1
			} else {
				width, height = cfg.Width, cfg.Height
			}
		} else {
			corrupt = 1
		}
	}

	status := domain.SampleOK
	if corrupt == 1 {
		status = domain.SampleCorrupt
	}
	rel := filepath.Join(sum[:2], fmt.Sprintf("%s_%s", sum[:8], sanitize(filename)))

	// Content-addressed storage under the dataset directory.
	abs, err := util.SafeJoin(filepath.Join(s.dataRoot, ds.ID), rel)
	if err != nil {
		rejected = 1
		errMsg = fmt.Sprintf("%s: %v", filename, err)
		return
	}
	if _, err := os.Stat(abs); err != nil {
		data, rerr := os.ReadFile(tmp.Name())
		if rerr != nil {
			errMsg = rerr.Error()
			return
		}
		if err := util.AtomicWrite(abs, data, 0o644); err != nil {
			errMsg = err.Error()
			return
		}
	}

	if existing != "" {
		// Already have identical content in this dataset: record a duplicate
		// marker sample so the user sees it, but keep it out of training.
		if corrupt == 0 {
			status = domain.SampleDuplicate
		}
		dup = 1
	} else {
		added = 1
	}

	meta := map[string]any{}
	_ = meta
	sm := &domain.Sample{
		ID: util.NewID(), DatasetID: ds.ID, Filename: filename, RelPath: rel,
		MediaType: mt, SizeBytes: n, SHA256: sum, Width: width, Height: height,
		Status: status, Meta: map[string]any{},
	}
	if classID != "" {
		v := classID
		sm.ClassID = &v
	}
	metaJSON, _ := json.Marshal(map[string]any{})
	defer s.st.Lock()()
	_, err = s.st.DB().ExecContext(ctx, `INSERT INTO samples
          (id, dataset_id, class_id, filename, rel_path, media_type, size_bytes, sha256,
           width, height, duration_ms, status, meta, created_at)
          VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)`,
		sm.ID, sm.DatasetID, sm.ClassID, sm.Filename, sm.RelPath, sm.MediaType, sm.SizeBytes, sm.SHA256,
		sm.Width, sm.Height, sm.DurationMS, sm.Status, string(metaJSON),
		timeNowRFC3339())
	if err != nil {
		errMsg = err.Error()
		added, dup = 0, 0
		return
	}
	return
}

func sanitize(name string) string {
	var b strings.Builder
	for _, r := range name {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9',
			r == '.', r == '-', r == '_':
			b.WriteRune(r)
		default:
			b.WriteRune('_')
		}
	}
	s := b.String()
	if len(s) > 80 {
		s = s[len(s)-80:]
	}
	return s
}

func timeNowRFC3339() string { return time.Now().UTC().Format(time.RFC3339) }

// ImportZIP extracts a zip archive into a dataset. Directory names inside the
// zip become class names (standard labeled-folder layout).
func (s *Service) ImportZIP(ctx context.Context, ds *domain.Dataset, r io.Reader, sizeLimit int64) (*ImportResult, error) {
	tmp, err := os.CreateTemp(s.dataRoot, ".zip-*")
	if err != nil {
		return nil, err
	}
	defer os.Remove(tmp.Name())
	n, err := io.Copy(tmp, io.LimitReader(r, sizeLimit))
	tmp.Close()
	if err != nil {
		return nil, util.ErrBadRequest("zip read error")
	}
	if n >= sizeLimit {
		return nil, util.ErrBadRequest("zip exceeds size limit")
	}
	res := &ImportResult{}
	zr, err := zip.OpenReader(tmp.Name())
	if err != nil {
		return nil, util.ErrBadRequest("invalid zip archive: %v", err)
	}
	defer zr.Close()
	for _, zf := range zr.File {
		name := filepath.ToSlash(zf.Name)
		if strings.Contains(name, "..") || strings.HasPrefix(name, "/") {
			res.Rejected++
			continue
		}
		base := filepath.Base(name)
		if base == "" || strings.HasPrefix(base, ".") || strings.HasPrefix(base, "__MACOSX") {
			continue
		}
		ext := strings.ToLower(filepath.Ext(base))
		if _, ok := allowedExt[ext]; !ok {
			continue
		}
		dir := strings.TrimSuffix(name, "/"+base)
		parts := strings.Split(dir, "/")
		classHint := ""
		for _, p := range parts {
			if p != "" && !strings.Contains(p, ".") && !strings.HasPrefix(p, "__MACOSX") {
				classHint = p
			}
		}
		rc, err := zf.Open()
		if err != nil {
			res.Errors = append(res.Errors, fmt.Sprintf("%s: %v", name, err))
			continue
		}
		classID := ""
		if classHint != "" {
			if c, err := s.AddClass(ctx, ds.ID, classHint); err == nil {
				classID = c.ID
			} else if util.APIErrCode(err) == "duplicate_class" {
				classes, _ := s.ListClasses(ctx, ds.ID)
				for _, cc := range classes {
					if cc.Name == classHint {
						classID = cc.ID
					}
				}
			}
		}
		a, d, c, rj, em := s.storeOne(ctx, ds, rc, base, classID)
		rc.Close()
		res.Added += a
		res.Duplicates += d
		res.Corrupt += c
		res.Rejected += rj
		if em != "" {
			res.Errors = append(res.Errors, em)
		}
	}
	s.touchDataset(ctx, ds.ID)
	return res, nil
}
