package ml

import (
	"image"
	"image/color"
	"image/draw"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"math/rand"
	"neurostudio/internal/domain"
)

// ImageToTensor decodes, resizes and normalizes an image into a batch-ready
// tensor [1, C, S, S] with values in [0,1].
func ImageToTensor(img image.Image, size int, grayscale bool) (*Tensor, error) {
	b := img.Bounds()
	if b.Dx() <= 0 || b.Dy() <= 0 {
		return nil, ErrShape("empty image")
	}
	resized := ResizeBilinear(img, size, size)
	ch := 3
	if grayscale {
		ch = 1
	}
	t := NewTensor(1, ch, size, size)
	grayT := func(x, y int) (r, g, bl float32) {
		c := resized.At(x, y)
		r32, g32, b32, _ := c.RGBA()
		return float32(r32 >> 8), float32(g32 >> 8), float32(b32 >> 8)
	}
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			r, g, bl := grayT(x, y)
			if grayscale {
				v := 0.299*r + 0.587*g + 0.114*bl
				t.SetAt(0, y, x, v/255)
			} else {
				t.SetAt(0, y, x, r/255)
				t.SetAt(1, y, x, g/255)
				t.SetAt(2, y, x, bl/255)
			}
		}
	}
	return t, nil
}

// ResizeBilinear scales src to w x h using bilinear interpolation.
func ResizeBilinear(src image.Image, w, h int) *image.RGBA {
	b := src.Bounds()
	out := image.NewRGBA(image.Rect(0, 0, w, h))
	draw.Draw(out, out.Bounds(), image.Transparent, image.Point{}, draw.Src)
	sw, sh := b.Dx(), b.Dy()
	if sw == 0 || sh == 0 {
		return out
	}
	xRatio := float64(sw) / float64(w)
	yRatio := float64(sh) / float64(h)
	for y := 0; y < h; y++ {
		sy := (float64(y)+0.5)*yRatio - 0.5
		y0 := int(math.Floor(sy))
		fy := sy - float64(y0)
		y0c, y1c := clampi(y0, 0, sh-1), clampi(y0+1, 0, sh-1)
		for x := 0; x < w; x++ {
			sx := (float64(x)+0.5)*xRatio - 0.5
			x0 := int(math.Floor(sx))
			fx := sx - float64(x0)
			x0c, x1c := clampi(x0, 0, sw-1), clampi(x0+1, 0, sw-1)
			c00 := rgbaF(src.At(b.Min.X+x0c, b.Min.Y+y0c))
			c10 := rgbaF(src.At(b.Min.X+x1c, b.Min.Y+y0c))
			c01 := rgbaF(src.At(b.Min.X+x0c, b.Min.Y+y1c))
			c11 := rgbaF(src.At(b.Min.X+x1c, b.Min.Y+y1c))
			out.SetRGBA(x, y, lerp4(c00, c10, c01, c11, fx, fy))
		}
	}
	return out
}

func clampi(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func rgbaF(c interface{ RGBA() (r, g, b, a uint32) }) [4]float64 {
	r, g, b, a := c.RGBA()
	return [4]float64{float64(r >> 8), float64(g >> 8), float64(b >> 8), float64(a >> 8)}
}

func lerp4(c00, c10, c01, c11 [4]float64, fx, fy float64) (out color.RGBA) {
	ch := [4]uint8{}
	for i := 0; i < 4; i++ {
		top := c00[i]*(1-fx) + c10[i]*fx
		bot := c01[i]*(1-fx) + c11[i]*fx
		v := top*(1-fy) + bot*fy
		ch[i] = uint8(clampi(int(math.Round(v)), 0, 255))
	}
	return color.RGBA{ch[0], ch[1], ch[2], ch[3]}
}

// Augment applies the configured, seeded augmentations to one tensor
// [1,C,H,W]. It is applied to training data only — the caller guarantees
// validation/test tensors are never passed here.
func Augment(t *Tensor, cfg domain.AugmentConfig, rng *rand.Rand) *Tensor {
	C, H, W := t.Shape[1], t.Shape[2], t.Shape[3]
	out := t.Clone()

	if cfg.CropPad > 0 && rng.Float64() < 0.75 {
		pad := cfg.CropPad
		padded := NewTensor(1, C, H+2*pad, W+2*pad)
		for c := 0; c < C; c++ {
			for y := 0; y < H; y++ {
				copy(padded.Data[((c*(H+2*pad)+y+pad)*(W+2*pad))+pad:], out.Data[(c*H+y)*W:(c*H+y)*W+W])
			}
		}
		oy, ox := rng.Intn(2*pad+1), rng.Intn(2*pad+1)
		for c := 0; c < C; c++ {
			for y := 0; y < H; y++ {
				copy(out.Data[(c*H+y)*W:(c*H+y)*W+W],
					padded.Data[((c*(H+2*pad)+y+oy)*(W+2*pad))+ox:])
			}
		}
	}

	if cfg.HFlip && rng.Float64() < 0.5 {
		for c := 0; c < C; c++ {
			for y := 0; y < H; y++ {
				for x := 0; x < W/2; x++ {
					i1 := (c*H+y)*W + x
					i2 := (c*H+y)*W + (W - 1 - x)
					out.Data[i1], out.Data[i2] = out.Data[i2], out.Data[i1]
				}
			}
		}
	}

	if cfg.Rotate && rng.Float64() < 0.4 {
		angle := (2*rng.Float64() - 1) * (cfg.MaxAngle * math.Pi / 180)
		out = rotateNearest(out, angle)
	}

	if cfg.Brightness && rng.Float64() < 0.5 {
		delta := float32(0.2 * (2*rng.Float64() - 1))
		for i := range out.Data {
			out.Data[i] = clampf(out.Data[i]+delta, 0, 1)
		}
	}
	if cfg.Contrast && rng.Float64() < 0.4 {
		factor := 1 + float32(0.3*(2*rng.Float64()-1))
		mean := float32(0)
		for _, v := range out.Data {
			mean += v
		}
		mean /= float32(len(out.Data))
		for i := range out.Data {
			out.Data[i] = clampf((out.Data[i]-mean)*factor+mean, 0, 1)
		}
	}
	if cfg.Noise && rng.Float64() < 0.3 {
		for i := range out.Data {
			out.Data[i] = clampf(out.Data[i]+float32(rng.NormFloat64())*0.03, 0, 1)
		}
	}
	return out
}

func rotateNearest(t *Tensor, angle float64) *Tensor {
	C, H, W := t.Shape[1], t.Shape[2], t.Shape[3]
	out := NewTensor(t.Shape...)
	cy, cx := float64(H)/2-0.5, float64(W)/2-0.5
	sin, cos := math.Sin(angle), math.Cos(angle)
	for c := 0; c < C; c++ {
		for y := 0; y < H; y++ {
			for x := 0; x < W; x++ {
				dx, dy := float64(x)-cx, float64(y)-cy
				// Inverse mapping: sample the source pixel that rotates
				// into (x, y). Out-of-bounds samples become 0 (black).
				sx := int(math.Round(cx + cos*dx - sin*dy))
				sy := int(math.Round(cy + sin*dx + cos*dy))
				if sx >= 0 && sx < W && sy >= 0 && sy < H {
					out.Data[(c*H+y)*W+x] = t.Data[(c*H+sy)*W+sx]
				}
			}
		}
	}
	return out
}

func clampf(v, lo, hi float32) float32 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
