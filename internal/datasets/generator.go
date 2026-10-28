package datasets

import (
	"bytes"
	"context"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"math"
	"math/rand"
	"strings"

	"neurostudio/internal/domain"
)

// shapeKinds are the synthetic classes the generator can draw. Each class is
// visually distinct and separable, which makes the sample dataset ideal for
// verifying the end-to-end training workflow on a laptop CPU.
var shapeKinds = []string{"circle", "square", "triangle", "star", "cross", "diamond"}

// GenerateSampleDataset creates a new dataset with procedurally drawn shape
// images. Deterministic given the seed.
func (s *Service) GenerateSampleDataset(ctx context.Context, projectID, name string, classes int, perClass int, seed int64) (*domain.Dataset, *ImportResult, error) {
	if classes < 2 || classes > len(shapeKinds) {
		classes = 3
	}
	if perClass < 10 || perClass > 500 {
		perClass = 40
	}
	ds, err := s.Create(ctx, projectID, CreateInput{
		Name: name, DataType: domain.DataImage,
		Notes: fmt.Sprintf("Synthetic shapes dataset (seed=%d, %d classes × %d samples). Generated locally — no external data.", seed, classes, perClass),
	})
	if err != nil {
		return nil, nil, err
	}
	rng := rand.New(rand.NewSource(seed))
	res := &ImportResult{}
	for ci := 0; ci < classes; ci++ {
		kind := shapeKinds[ci%len(shapeKinds)]
		c, err := s.AddClass(ctx, ds.ID, kind)
		if err != nil {
			return ds, res, err
		}
		for i := 0; i < perClass; i++ {
			img := drawShape(rng, kind, 96)
			var buf bytes.Buffer
			if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: 88}); err != nil {
				return ds, res, err
			}
			fname := fmt.Sprintf("%s_%03d.jpg", kind, i)
			a, d, c, rj, em := s.storeOne(ctx, ds, bytes.NewReader(buf.Bytes()), fname, c.ID)
			res.Added += a
			res.Duplicates += d
			res.Corrupt += c
			res.Rejected += rj
			if em != "" {
				res.Errors = append(res.Errors, em)
			}
		}
		res.Classes = append(res.Classes, kind)
	}
	s.touchDataset(ctx, ds.ID)
	return ds, res, nil
}

// drawShape renders one noisy 96x96 JPEG-able image with the given shape.
func drawShape(rng *rand.Rand, kind string, size int) image.Image {
	img := image.NewRGBA(image.Rect(0, 0, size, size))

	// Random background gradient-ish base color.
	base := color.RGBA{
		uint8(40 + rng.Intn(160)),
		uint8(40 + rng.Intn(160)),
		uint8(40 + rng.Intn(160)),
		255,
	}
	draw.Draw(img, img.Bounds(), &image.Uniform{base}, image.Point{}, draw.Src)

	// Slight vertical gradient for realism.
	top := color.RGBA{uint8(int(base.R) + rng.Intn(30)), uint8(int(base.G) + rng.Intn(30)), uint8(int(base.B) + rng.Intn(30)), 255}
	for y := 0; y < size/2; y++ {
		for x := 0; x < size; x++ {
			img.Set(x, y, top)
		}
	}

	// Shape color contrasts with background.
	fg := color.RGBA{
		uint8(200 + rng.Intn(55)),
		uint8(200 + rng.Intn(55)),
		uint8(200 + rng.Intn(55)),
		255,
	}

	cx := float64(size)/2 + float64(rng.Intn(8)-4)
	cy := float64(size)/2 + float64(rng.Intn(8)-4)
	r := float64(size)/3 + float64(rng.Intn(8)-4)

	switch kind {
	case "circle":
		drawDisk(img, cx, cy, r, fg)
	case "square":
		half := r * 0.85
		x0, y0 := int(cx-half), int(cy-half)
		x1, y1 := int(cx+half), int(cy+half)
		for y := y0; y <= y1; y++ {
			for x := x0; x <= x1; x++ {
				if inBounds(img, x, y) {
					img.Set(x, y, fg)
				}
			}
		}
	case "triangle":
		drawPolygon(img, []pt{{cx, cy - r}, {cx - r, cy + r}, {cx + r, cy + r}}, fg)
	case "star":
		drawPolygon(img, starPoints(cx, cy, r, 5), fg)
	case "cross":
		w := r * 0.35
		for y := int(cy - r); y <= int(cy+r); y++ {
			for x := int(cx - w); x <= int(cx+w); x++ {
				if inBounds(img, x, y) {
					img.Set(x, y, fg)
				}
			}
		}
		for y := int(cy - w); y <= int(cy+w); y++ {
			for x := int(cx - r); x <= int(cx+r); x++ {
				if inBounds(img, x, y) {
					img.Set(x, y, fg)
				}
			}
		}
	case "diamond":
		drawPolygon(img, []pt{{cx, cy - r}, {cx + r, cy}, {cx, cy + r}, {cx - r, cy}}, fg)
	}

	// Speckle noise.
	for i := 0; i < 400; i++ {
		x, y := rng.Intn(size), rng.Intn(size)
		v := uint8(rng.Intn(255))
		img.Set(x, y, color.RGBA{v, v, v, 255})
	}
	return img
}

type pt struct{ x, y float64 }

func starPoints(cx, cy, r float64, n int) []pt {
	pts := make([]pt, 0, n*2)
	for i := 0; i < n*2; i++ {
		ang := math.Pi*float64(i)/float64(n) - math.Pi/2
		rad := r
		if i%2 == 1 {
			rad = r * 0.45
		}
		pts = append(pts, pt{cx + rad*math.Cos(ang), cy + rad*math.Sin(ang)})
	}
	return pts
}

func drawDisk(img *image.RGBA, cx, cy, r float64, c color.RGBA) {
	for y := int(cy - r); y <= int(cy+r); y++ {
		for x := int(cx - r); x <= int(cx+r); x++ {
			dx, dy := float64(x)-cx, float64(y)-cy
			if dx*dx+dy*dy <= r*r && inBounds(img, x, y) {
				img.Set(x, y, c)
			}
		}
	}
}

func drawPolygon(img *image.RGBA, poly []pt, c color.RGBA) {
	minY, maxY := math.MaxFloat64, -math.MaxFloat64
	for _, p := range poly {
		minY = math.Min(minY, p.y)
		maxY = math.Max(maxY, p.y)
	}
	for y := int(minY); y <= int(maxY); y++ {
		xs := []float64{}
		j := len(poly) - 1
		for i := 0; i < len(poly); i++ {
			a, b := poly[i], poly[j]
			if (a.y <= float64(y) && b.y > float64(y)) || (b.y <= float64(y) && a.y > float64(y)) {
				xs = append(xs, a.x+(float64(y)-a.y)/(b.y-a.y)*(b.x-a.x))
			}
			j = i
		}
		for k := 0; k+1 < len(xs); k += 2 {
			x0, x1 := xs[k], xs[k+1]
			if x0 > x1 {
				x0, x1 = x1, x0
			}
			for x := int(x0); x <= int(x1); x++ {
				if inBounds(img, x, y) {
					img.Set(x, y, c)
				}
			}
		}
	}
}

func inBounds(img *image.RGBA, x, y int) bool {
	return x >= 0 && y >= 0 && x < img.Bounds().Dx() && y < img.Bounds().Dy()
}

var _ = strings.TrimSpace
