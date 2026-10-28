// Package ml implements the real mathematics behind the native training
// backend: tensors, convolution, dense layers, optimizers and metrics.
// Every operation has a hand-written backward pass validated by numerical
// gradient-check tests. No external ML library is used; the compute is
// genuinely performed on the CPU by this code.
package ml

import (
	"math"
	"math/rand"
	"sync"
)

// Tensor is a dense float32 tensor in row-major layout.
// For images the layout is channel-major: [C][H][W] flattened.
type Tensor struct {
	Shape []int
	Data  []float32
}

func NewTensor(shape ...int) *Tensor {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return &Tensor{Shape: append([]int(nil), shape...), Data: make([]float32, n)}
}

func (t *Tensor) Size() int {
	n := 1
	for _, d := range t.Shape {
		n *= d
	}
	return n
}

func (t *Tensor) Clone() *Tensor {
	return &Tensor{Shape: append([]int(nil), t.Shape...), Data: append([]float32(nil), t.Data...)}
}

func (t *Tensor) Zero() {
	for i := range t.Data {
		t.Data[i] = 0
	}
}

// HeInit fills with He-normal values (for ReLU networks).
func (t *Tensor) HeInit(rng *rand.Rand, fanIn int) {
	std := float32(math.Sqrt(2.0 / float64(fanIn)))
	for i := range t.Data {
		t.Data[i] = float32(rng.NormFloat64()) * std
	}
}

// ---- image helpers on channel-major tensors ----

func (t *Tensor) At(c, y, x int) float32 {
	_, h, w := t.Shape[0], t.Shape[1], t.Shape[2]
	return t.Data[(c*h+y)*w+x]
}

func (t *Tensor) SetAt(c, y, x int, v float32) {
	_, h, w := t.Shape[0], t.Shape[1], t.Shape[2]
	t.Data[(c*h+y)*w+x] = v
}

// ---- Conv2D ----

type Conv2D struct {
	Filters, Channels, K, Stride, Pad int
	W                                 *Tensor // [F][C][K][K]
	B                                 *Tensor // [F]
}

func NewConv2D(filters, channels, k, stride, pad int, rng *rand.Rand) *Conv2D {
	c := &Conv2D{
		Filters: filters, Channels: channels, K: k, Stride: stride, Pad: pad,
		W: NewTensor(filters, channels, k, k),
		B: NewTensor(filters),
	}
	c.W.HeInit(rng, channels*k*k)
	return c
}

// Forward computes x[N,C,H,W] -> y[N,F,H',W'] in parallel over (n,f).
func (c *Conv2D) Forward(x *Tensor, n int, workers int) *Tensor {
	C, H, W := x.Shape[1], x.Shape[2], x.Shape[3]
	oh := (H+2*c.Pad-c.K)/c.Stride + 1
	ow := (W+2*c.Pad-c.K)/c.Stride + 1
	y := NewTensor(n, c.Filters, oh, ow)

	type task struct{ ni, fi int }
	tasks := make(chan task, n*c.Filters)
	for ni := 0; ni < n; ni++ {
		for fi := 0; fi < c.Filters; fi++ {
			tasks <- task{ni, fi}
		}
	}
	close(tasks)

	if workers < 1 {
		workers = 1
	}
	var wg sync.WaitGroup
	for wk := 0; wk < workers; wk++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for t := range tasks {
				c.convOne(x, y, t.ni, t.fi, C, H, W, oh, ow)
			}
		}()
	}
	wg.Wait()
	return y
}

func (c *Conv2D) convOne(x, y *Tensor, ni, fi, C, H, W, oh, ow int) {
	for oy := 0; oy < oh; oy++ {
		for ox := 0; ox < ow; ox++ {
			var sum float32 = c.B.Data[fi]
			for ci := 0; ci < C; ci++ {
				for ky := 0; ky < c.K; ky++ {
					iy := oy*c.Stride + ky - c.Pad
					if iy < 0 || iy >= H {
						continue
					}
					for kx := 0; kx < c.K; kx++ {
						ix := ox*c.Stride + kx - c.Pad
						if ix < 0 || ix >= W {
							continue
						}
						wv := c.W.Data[((fi*C+ci)*c.K+ky)*c.K+kx]
						sum += wv * x.Data[((ni*C+ci)*H+iy)*W+ix]
					}
				}
			}
			y.Data[((ni*c.Filters+fi)*oh+oy)*ow+ox] = sum
		}
	}
}

// Backward accumulates parameter gradients and returns input gradient.
func (c *Conv2D) Backward(x, dOut *Tensor, n int, dW, dB *Tensor, workers int) *Tensor {
	C, H, W := x.Shape[1], x.Shape[2], x.Shape[3]
	_, F, oh, ow := dOut.Shape[0], dOut.Shape[1], dOut.Shape[2], dOut.Shape[3]
	dIn := NewTensor(x.Shape...)

	type task struct{ ni int }
	tasks := make(chan task, n)
	for ni := 0; ni < n; ni++ {
		tasks <- task{ni}
	}
	close(tasks)

	dWLocal := make([]*Tensor, workers)
	dBLocal := make([]*Tensor, workers)
	dInLocal := make([]*Tensor, workers)
	var wg sync.WaitGroup
	if workers < 1 {
		workers = 1
	}
	for wk := 0; wk < workers; wk++ {
		dWLocal[wk] = NewTensor(dW.Shape...)
		dBLocal[wk] = NewTensor(dB.Shape...)
		dInLocal[wk] = NewTensor(dIn.Shape...)
	}
	for wk := 0; wk < workers; wk++ {
		wg.Add(1)
		go func(wid int) {
			defer wg.Done()
			for t := range tasks {
				c.backOne(x, dOut, dInLocal[wid], dWLocal[wid], dBLocal[wid], t.ni, C, H, W, F, oh, ow)
			}
		}(wk)
	}
	wg.Wait()

	for wk := 0; wk < workers; wk++ {
		for i := range dW.Data {
			dW.Data[i] += dWLocal[wk].Data[i]
		}
		for i := range dB.Data {
			dB.Data[i] += dBLocal[wk].Data[i]
		}
		for i := range dIn.Data {
			dIn.Data[i] += dInLocal[wk].Data[i]
		}
	}
	return dIn
}

func (c *Conv2D) backOne(x, dOut, dIn, dW, dB *Tensor, ni, C, H, W, F, oh, ow int) {
	for fi := 0; fi < F; fi++ {
		for oy := 0; oy < oh; oy++ {
			for ox := 0; ox < ow; ox++ {
				d := dOut.Data[((ni*F+fi)*oh+oy)*ow+ox]
				if d == 0 {
					continue
				}
				dB.Data[fi] += d
				for ci := 0; ci < C; ci++ {
					for ky := 0; ky < c.K; ky++ {
						iy := oy*c.Stride + ky - c.Pad
						if iy < 0 || iy >= H {
							continue
						}
						for kx := 0; kx < c.K; kx++ {
							ix := ox*c.Stride + kx - c.Pad
							if ix < 0 || ix >= W {
								continue
							}
							dW.Data[((fi*C+ci)*c.K+ky)*c.K+kx] += d * x.Data[((ni*C+ci)*H+iy)*W+ix]
							dIn.Data[((ni*C+ci)*H+iy)*W+ix] += d * c.W.Data[((fi*C+ci)*c.K+ky)*c.K+kx]
						}
					}
				}
			}
		}
	}
}

// ---- ReLU ----

func ReLU(x *Tensor) *Tensor {
	y := x.Clone()
	for i, v := range y.Data {
		if v < 0 {
			y.Data[i] = 0
		}
	}
	return y
}

func ReLUBackward(y, dOut *Tensor) *Tensor {
	dIn := dOut.Clone()
	for i, v := range y.Data {
		if v <= 0 {
			dIn.Data[i] = 0
		}
	}
	return dIn
}

// ---- MaxPool 2x2 stride 2 ----

type MaxPool2 struct{}

func (m MaxPool2) OutDim(h, w int) (int, int) { return h / 2, w / 2 }

func (m MaxPool2) Forward(x *Tensor, n int) (*Tensor, []MaxArg) {
	C, H, W := x.Shape[1], x.Shape[2], x.Shape[3]
	oh, ow := H/2, W/2
	y := NewTensor(n, C, oh, ow)
	args := make([]MaxArg, n*C*oh*ow)
	for ni := 0; ni < n; ni++ {
		for ci := 0; ci < C; ci++ {
			for py := 0; py < oh; py++ {
				for px := 0; px < ow; px++ {
					y0 := x.At2(ni, ci, py*2, px*2)
					y1 := x.At2(ni, ci, py*2, px*2+1)
					y2 := x.At2(ni, ci, py*2+1, px*2)
					y3 := x.At2(ni, ci, py*2+1, px*2+1)
					best, idx := y0, 0
					if y1 > best {
						best, idx = y1, 1
					}
					if y2 > best {
						best, idx = y2, 2
					}
					if y3 > best {
						best, idx = y3, 3
					}
					y.Data[((ni*C+ci)*oh+py)*ow+px] = best
					args[(ni*C+ci)*oh*ow+py*ow+px] = MaxArg{Idx: idx}
				}
			}
		}
	}
	return y, args
}

type MaxArg struct{ Idx int }

func (t *Tensor) At2(n, c, y, x int) float32 {
	_, C, H, W := t.Shape[0], t.Shape[1], t.Shape[2], t.Shape[3]
	return t.Data[((n*C+c)*H+y)*W+x]
}

func MaxPool2Backward(x, dOut *Tensor, args []MaxArg, n int) *Tensor {
	C, H, W := x.Shape[1], x.Shape[2], x.Shape[3]
	oh, ow := H/2, W/2
	dIn := NewTensor(x.Shape...)
	for ni := 0; ni < n; ni++ {
		for ci := 0; ci < C; ci++ {
			for py := 0; py < oh; py++ {
				for px := 0; px < ow; px++ {
					d := dOut.Data[((ni*C+ci)*oh+py)*ow+px]
					arg := args[(ni*C+ci)*oh*ow+py*ow+px]
					dy, dx := arg.Idx/2, arg.Idx%2
					dIn.Data[((ni*C+ci)*H+py*2+dy)*W+px*2+dx] += d
				}
			}
		}
	}
	return dIn
}

// ---- Flatten + Dense ----

func FlattenBatch(x *Tensor, n int) *Tensor {
	C, H, W := x.Shape[1], x.Shape[2], x.Shape[3]
	return &Tensor{Shape: []int{n, C * H * W}, Data: x.Data}
}

type Dense struct {
	In, Out int
	W       *Tensor // [Out][In]
	B       *Tensor // [Out]
}

func NewDense(in, out int, rng *rand.Rand) *Dense {
	d := &Dense{In: in, Out: out, W: NewTensor(out, in), B: NewTensor(out)}
	// Glorot uniform.
	limit := float32(math.Sqrt(6.0 / float64(in+out)))
	for i := range d.W.Data {
		d.W.Data[i] = (float32(rng.Float64())*2 - 1) * limit
	}
	return d
}

func (d *Dense) Forward(x *Tensor, n int) *Tensor {
	y := NewTensor(n, d.Out)
	for ni := 0; ni < n; ni++ {
		row := x.Data[ni*d.In : (ni+1)*d.In]
		for o := 0; o < d.Out; o++ {
			var sum float32 = d.B.Data[o]
			wrow := d.W.Data[o*d.In : (o+1)*d.In]
			for i, v := range row {
				sum += wrow[i] * v
			}
			y.Data[ni*d.Out+o] = sum
		}
	}
	return y
}

// DenseBackward: dW [Out,In], dB [Out]; returns dIn.
func (d *Dense) Backward(x *Tensor, dOut *Tensor, n int) (dW, dB, dIn *Tensor) {
	dW = NewTensor(d.Out, d.In)
	dB = NewTensor(d.Out)
	dIn = NewTensor(n, d.In)
	for ni := 0; ni < n; ni++ {
		for o := 0; o < d.Out; o++ {
			dv := dOut.Data[ni*d.Out+o]
			dB.Data[o] += dv
			wrow := d.W.Data[o*d.In : (o+1)*d.In]
			diRow := dIn.Data[ni*d.In : (ni+1)*d.In]
			xRow := x.Data[ni*d.In : (ni+1)*d.In]
			for i := 0; i < d.In; i++ {
				dW.Data[o*d.In+i] += dv * xRow[i]
				diRow[i] += dv * wrow[i]
			}
		}
	}
	return
}

// ---- Softmax + CrossEntropy ----

// SoftmaxCE computes softmax, mean cross-entropy loss and dLogits.
func SoftmaxCE(logits *Tensor, labels []int, n, classes int) (loss float64, dLogits *Tensor, probs []float32) {
	probs = make([]float32, n*classes)
	dLogits = NewTensor(n, classes)
	total := 0.0
	for ni := 0; ni < n; ni++ {
		row := logits.Data[ni*classes : (ni+1)*classes]
		maxV := row[0]
		for _, v := range row[1:] {
			if v > maxV {
				maxV = v
			}
		}
		sum := 0.0
		exp := make([]float64, classes)
		for i, v := range row {
			e := math.Exp(float64(v - maxV))
			exp[i] = e
			sum += e
		}
		p := make([]float32, classes)
		for i := range exp {
			p[i] = float32(exp[i] / sum)
		}
		copy(probs[ni*classes:(ni+1)*classes], p)
		total += -math.Log(math.Max(float64(p[labels[ni]]), 1e-12))
		for i := 0; i < classes; i++ {
			g := p[i]
			if i == labels[ni] {
				g -= 1
			}
			dLogits.Data[ni*classes+i] = g / float32(n) // mean over batch
		}
	}
	return total / float64(n), dLogits, probs
}

// Argmax returns the index of the maximum value in a row.
func Argmax(probs []float32, classes, ni int) int {
	best, idx := float32(math.Inf(-1)), 0
	for i := 0; i < classes; i++ {
		if v := probs[ni*classes+i]; v > best {
			best, idx = v, i
		}
	}
	return idx
}
