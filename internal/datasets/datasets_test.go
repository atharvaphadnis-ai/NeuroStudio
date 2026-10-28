package datasets

import (
	"bytes"
	"context"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"testing"

	"neurostudio/internal/domain"
	"neurostudio/internal/store"
)

func testService(t *testing.T) (*Service, *store.Store) {
	t.Helper()
	dir := t.TempDir()
	st, err := store.Open(filepath.Join(dir, "test.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { st.Close() })
	_, err = st.DB().Exec(`INSERT INTO projects (id, name, task_type, created_at, updated_at) VALUES ('p1','T','image_classification','2024-01-01T00:00:00Z','2024-01-01T00:00:00Z')`)
	if err != nil {
		t.Fatal(err)
	}
	return NewService(st, filepath.Join(dir, "datasets")), st
}

func pngBytes(t *testing.T, w, h int, c color.RGBA) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, c)
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func TestImportDedupeAndCorrupt(t *testing.T) {
	svc, _ := testService(t)
	ctx := context.Background()
	ds, err := svc.Create(ctx, "p1", CreateInput{Name: "d1", DataType: domain.DataImage})
	if err != nil {
		t.Fatal(err)
	}
	c1, _ := svc.AddClass(ctx, ds.ID, "red")

	// Valid image.
	img := pngBytes(t, 8, 8, color.RGBA{255, 0, 0, 255})
	a, dup, corrupt, rej, _ := svc.storeOne(ctx, ds, bytes.NewReader(img), "a.png", c1.ID)
	if a != 1 || dup != 0 || corrupt != 0 || rej != 0 {
		t.Fatalf("first import: added=%d dup=%d corrupt=%d rej=%d", a, dup, corrupt, rej)
	}
	// Same content, different name -> duplicate detection.
	_, dup, _, _, _ = svc.storeOne(ctx, ds, bytes.NewReader(img), "b.png", c1.ID)
	if dup != 1 {
		t.Fatalf("duplicate not detected")
	}
	// Corrupt content with image extension.
	a, _, corrupt, _, _ = svc.storeOne(ctx, ds, bytes.NewReader([]byte("not an image")), "bad.png", c1.ID)
	if corrupt != 1 {
		t.Fatalf("corrupt not detected (added=%d)", a)
	}
	// Unsupported extension rejected.
	_, _, _, rej, _ = svc.storeOne(ctx, ds, bytes.NewReader([]byte("<script>")), "evil.html", "")
	if rej != 1 {
		t.Fatalf("unsupported extension not rejected")
	}

	stats, err := svc.Stats(ctx, ds.ID)
	if err != nil {
		t.Fatal(err)
	}
	if stats.ByStatus["duplicate"] != 1 || stats.ByStatus["corrupt"] != 1 {
		t.Fatalf("stats wrong: %+v", stats.ByStatus)
	}
}

func TestPathTraversalRejected(t *testing.T) {
	svc, _ := testService(t)
	if _, err := svc.SampleFile("ds1", "../../etc/passwd"); err == nil {
		t.Fatal("traversal accepted")
	}
	if _, err := svc.SampleFile("ds1", "/etc/passwd"); err == nil {
		t.Fatal("absolute path accepted")
	}
	if p, err := svc.SampleFile("ds1", "aa/abc123_file.png"); err != nil || p == "" {
		t.Fatalf("valid path rejected: %v %v", p, err)
	}
}

func TestSplitDeterminismAndLeakage(t *testing.T) {
	svc, st := testService(t)
	ctx := context.Background()
	ds, _ := svc.Create(ctx, "p1", CreateInput{Name: "d2", DataType: domain.DataImage})
	ca, _ := svc.AddClass(ctx, ds.ID, "one")
	cb, _ := svc.AddClass(ctx, ds.ID, "two")

	// 30 samples per class; class "two" includes duplicate content (leakage).
	imgA := pngBytes(t, 8, 8, color.RGBA{0, 255, 0, 255})
	for i := 0; i < 30; i++ {
		data := pngBytes(t, 8+i, 8, color.RGBA{uint8(i * 7), 0, 0, 255})
		svc.storeOne(ctx, ds, bytes.NewReader(data), "a"+string(rune('a'+i%26))+string(rune('0'+i/26))+".png", ca.ID)
	}
	for i := 0; i < 30; i++ {
		var data []byte
		if i < 6 {
			data = imgA // identical content to class one sample 0..5? make own dup content
		} else {
			data = pngBytes(t, 9+i, 9, color.RGBA{0, uint8(i * 7), 0, 255})
		}
		svc.storeOne(ctx, ds, bytes.NewReader(data), "b"+string(rune('a'+i%26))+string(rune('0'+i/26))+".png", cb.ID)
	}

	params := domain.SplitParams{Train: 0.7, Val: 0.15, Test: 0.15, Seed: 42}
	v1, err := svc.CreateVersion(ctx, ds.ID, params)
	if err != nil {
		t.Fatal(err)
	}
	v2, err := svc.CreateVersion(ctx, ds.ID, params)
	if err != nil {
		t.Fatal(err)
	}
	// Same seed => identical splits (reproducibility).
	if len(v1.Splits) != len(v2.Splits) {
		t.Fatalf("split sizes differ: %d vs %d", len(v1.Splits), len(v2.Splits))
	}
	for k, v := range v1.Splits {
		if v2.Splits[k] != v {
			t.Fatalf("split for %s differs between runs with same seed", k)
		}
	}
	// Rough fraction check.
	var train, val, test int
	for _, v := range v1.Splits {
		switch v {
		case "train":
			train++
		case "val":
			val++
		case "test":
			test++
		}
	}
	total := len(v1.Splits)
	if train*100 < total*60 || train*100 > total*80 {
		t.Fatalf("train fraction off: %d/%d", train, total)
	}
	if val == 0 || test == 0 {
		t.Fatalf("val/test empty: %d %d", val, test)
	}
	// Duplicates and corrupt are excluded.
	// Duplicates and corrupt are excluded: class "b" reused imgA for 6
	// samples; the first became ok, 5 were flagged duplicate -> 55 usable.
	if total != 55 {
		t.Fatalf("expected ok samples only in splits, got %d", total)
	}

	// Leakage detector: simulate an import path that bypassed dedupe by
	// re-activating the duplicate samples, then split again — the identical
	// hashes must produce a leakage warning.
	if _, err := st.DB().ExecContext(ctx, `UPDATE samples SET status='ok' WHERE status='duplicate'`); err != nil {
		t.Fatal(err)
	}
	v4, err := svc.CreateVersion(ctx, ds.ID, params)
	if err != nil {
		t.Fatal(err)
	}
	leaky := false
	for _, w := range v4.Stats.Warnings {
		if bytes.Contains([]byte(w), []byte("leakage")) {
			leaky = true
		}
	}
	if !leaky {
		t.Fatalf("leakage not detected: %+v", v4.Stats.Warnings)
	}

	// Different seed => different assignment (statistically certain here).
	v3, err := svc.CreateVersion(ctx, ds.ID, domain.SplitParams{Train: 0.7, Val: 0.15, Test: 0.15, Seed: 7})
	if err != nil {
		t.Fatal(err)
	}
	same := true
	for k, v := range v1.Splits {
		if v3.Splits[k] != v {
			same = false
			break
		}
	}
	if same {
		t.Fatal("different seed produced identical splits")
	}
}

func TestManifestRoundTrip(t *testing.T) {
	svc, _ := testService(t)
	ctx := context.Background()
	ds, _ := svc.Create(ctx, "p1", CreateInput{Name: "d3", DataType: domain.DataImage})
	ca, _ := svc.AddClass(ctx, ds.ID, "x")
	cb, _ := svc.AddClass(ctx, ds.ID, "y")
	for i := 0; i < 20; i++ {
		svc.storeOne(ctx, ds, bytes.NewReader(pngBytes(t, 5+i, 5, color.RGBA{255, 255, 0, 255})), "x"+string(rune('A'+i))+".png", ca.ID)
		svc.storeOne(ctx, ds, bytes.NewReader(pngBytes(t, 5, 5+i, color.RGBA{0, 255, 255, 255})), "y"+string(rune('A'+i))+".png", cb.ID)
	}
	dv, err := svc.CreateVersion(ctx, ds.ID, domain.SplitParams{Train: 0.7, Val: 0.15, Test: 0.15, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	m, err := svc.LoadManifest(dv)
	if err != nil {
		t.Fatal(err)
	}
	if len(m.Classes) != 2 || len(m.Samples) != 40 {
		t.Fatalf("manifest wrong: classes=%v samples=%d", m.Classes, len(m.Samples))
	}
	if m.Format != "neurostudio/manifest-1" {
		t.Fatalf("format=%s", m.Format)
	}
	// All samples must have a split and a class.
	for _, s := range m.Samples {
		if s.Split == "" || s.Class == "" {
			t.Fatalf("sample without split/class: %+v", s)
		}
	}
}

func TestGeneratorProducesTrainableDataset(t *testing.T) {
	svc, _ := testService(t)
	ctx := context.Background()
	ds, res, err := svc.GenerateSampleDataset(ctx, "p1", "gen", 3, 12, 42)
	if err != nil {
		t.Fatal(err)
	}
	if res.Added != 36 {
		t.Fatalf("added=%d want 36", res.Added)
	}
	if res.Corrupt != 0 {
		t.Fatalf("generator produced corrupt files")
	}
	dv, err := svc.CreateVersion(ctx, ds.ID, domain.SplitParams{Train: 0.7, Val: 0.15, Test: 0.15, Seed: 42})
	if err != nil {
		t.Fatal(err)
	}
	if len(dv.Splits) != 36 {
		t.Fatalf("splits=%d want 36", len(dv.Splits))
	}
	// Files must exist on disk.
	for _, sm := range dv.Splits {
		_ = sm
		break
	}
	samples, total, err := svc.ListSamples(ctx, ds.ID, "", "", 0, 100)
	if err != nil || total != 36 {
		t.Fatalf("list samples: %d %v", total, err)
	}
	for _, sm := range samples {
		p, err := svc.SampleFile(sm.DatasetID, sm.RelPath)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := os.Stat(p); err != nil {
			t.Fatalf("sample file missing: %s", p)
		}
	}
}
