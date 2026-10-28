package ml

import (
	"math"
	"math/rand"
	"testing"
)

// finiteDiff estimates d loss / d param numerically for gradient checking.
func finiteDiff(f func() float64, p *Tensor, idx int, eps float32) float64 {
	orig := p.Data[idx]
	p.Data[idx] = orig + eps
	lp := f()
	p.Data[idx] = orig - eps
	lm := f()
	p.Data[idx] = orig
	return float64(lp-lm) / float64(2*eps)
}

func relErr(a, b float64) float64 {
	d := math.Abs(a - b)
	m := math.Max(math.Abs(a), math.Abs(b))
	if m == 0 {
		return d
	}
	return d / m
}

func randInput(rng *rand.Rand, n, c, h, w int) *Tensor {
	x := NewTensor(n, c, h, w)
	for i := range x.Data {
		x.Data[i] = float32(rng.NormFloat64())
	}
	return x
}

// TestGradientCheckConv validates the Conv2D backward pass against finite
// differences on a tiny configuration.
func TestGradientCheckConv(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	conv := NewConv2D(2, 2, 3, 1, 1, rng)
	n := 2
	x := randInput(rng, n, 2, 6, 6)
	labels := []int{0, 1}
	// 6x6 -> conv pad1 -> 6x6 -> maxpool2 -> 3x3; flatten = 2*3*3 = 18
	dense := NewDense(2*3*3, 2, rng)

	lossFn := func() float64 {
		y := conv.Forward(x, n, 1)
		h := ReLU(y)
		p, _ := MaxPool2{}.Forward(h, n)
		flat := FlattenBatch(p, n)
		logits := dense.Forward(flat, n)
		l, _, _ := SoftmaxCE(logits, labels, n, 2)
		return l
	}

	y := conv.Forward(x, n, 1)
	h := ReLU(y)
	p, args := MaxPool2{}.Forward(h, n)
	flat := FlattenBatch(p, n)
	logits := dense.Forward(flat, n)
	_, dLogits, _ := SoftmaxCE(logits, labels, n, 2)
	dFlat := NewTensor(flat.Shape...)
	_, _, dInDense := dense.Backward(flat, dLogits, n)
	copy(dFlat.Data, dInDense.Data)
	dP := &Tensor{Shape: p.Shape, Data: dFlat.Data}
	dH := MaxPool2Backward(h, dP, args, n)
	dZ := ReLUBackward(h, dH)
	dW := NewTensor(conv.W.Shape...)
	dB := NewTensor(conv.B.Shape...)
	conv.Backward(x, dZ, n, dW, dB, 1)

	// Check a sample of weight and bias gradients.
	checks := 0
	for _, pi := range []int{0, 5, 17, 23, 40} {
		if pi >= len(conv.W.Data) {
			continue
		}
		num := finiteDiff(lossFn, conv.W, pi, 1e-3)
		ana := float64(dW.Data[pi])
		if e := relErr(num, ana); e > 0.05 {
			t.Fatalf("conv W grad mismatch idx=%d numeric=%v analytic=%v relerr=%v", pi, num, ana, e)
		}
		checks++
	}
	num := finiteDiff(lossFn, conv.B, 1, 1e-3)
	if e := relErr(num, float64(dB.Data[1])); e > 0.05 {
		t.Fatalf("conv B grad mismatch numeric=%v analytic=%v", num, dB.Data[1])
	}
	if checks == 0 {
		t.Fatal("no gradient checks ran")
	}
}

// TestGradientCheckDense validates the Dense backward pass.
func TestGradientCheckDense(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	dense := NewDense(8, 3, rng)
	n := 2
	x := randInput(rng, n, 1, 1, 8)
	flat := &Tensor{Shape: []int{n, 8}, Data: x.Data}
	labels := []int{1, 2}

	lossFn := func() float64 {
		logits := dense.Forward(flat, n)
		l, _, _ := SoftmaxCE(logits, labels, n, 3)
		return l
	}
	logits := dense.Forward(flat, n)
	_, dLogits, _ := SoftmaxCE(logits, labels, n, 3)
	dW, dB, _ := dense.Backward(flat, dLogits, n)

	for _, pi := range []int{0, 3, 10, 23} {
		num := finiteDiff(lossFn, dense.W, pi, 1e-3)
		if e := relErr(num, float64(dW.Data[pi])); e > 0.05 {
			t.Fatalf("dense W grad mismatch idx=%d numeric=%v analytic=%v", pi, num, dW.Data[pi])
		}
	}
	num := finiteDiff(lossFn, dense.B, 2, 1e-3)
	if e := relErr(num, float64(dB.Data[2])); e > 0.05 {
		t.Fatalf("dense B grad mismatch numeric=%v analytic=%v", num, dB.Data[2])
	}
}

// TestSGDDecreasesLoss proves the optimizer actually learns: a linearly
// separable 2-class problem must be driven to ~zero loss by gradient descent.
func TestSGDDecreasesLoss(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	dense := NewDense(2, 2, rng)
	// Data: class 0 near (1,0), class 1 near (-1,0).
	const n = 8
	x := NewTensor(n, 1, 1, 2)
	labels := make([]int, n)
	for i := 0; i < n; i++ {
		if i%2 == 0 {
			x.Data[i*2] = 1
			x.Data[i*2+1] = 0.1
			labels[i] = 0
		} else {
			x.Data[i*2] = -1
			x.Data[i*2+1] = -0.1
			labels[i] = 1
		}
	}
	opt := NewSGD(0.5, 0.9, 0)
	first := 0.0
	last := 0.0
	for step := 0; step < 200; step++ {
		logits := dense.Forward(x, n)
		loss, dL, _ := SoftmaxCE(logits, labels, n, 2)
		dW, dB, _ := dense.Backward(x, dL, n)
		opt.Step([]*Tensor{dense.W, dense.B}, []*Tensor{dW, dB})
		if step == 0 {
			first = loss
		}
		last = loss
	}
	if last > first {
		t.Fatalf("loss increased: first=%v last=%v", first, last)
	}
	if last > 0.05 {
		t.Fatalf("linearly separable problem not solved: loss=%v", last)
	}
}

// TestAdamLearnsXOR validates Adam on a tiny non-linear problem via the CNN
// stack (2x2 input, one conv + dense) — end-to-end numerical learning proof.
func TestAdamLearnsXOR(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	m := NewCNN(16, 1, 2, 0, 1, rng)
	// Four patterns: left-half bright vs right-half bright.
	patterns := []func(*Tensor){
		func(t *Tensor) { fillHalf(t, true) },
		func(t *Tensor) { fillHalf(t, false) },
	}
	const n = 4
	x := NewTensor(n, 1, 16, 16)
	labels := []int{0, 1, 0, 1}
	for i := 0; i < n; i++ {
		patterns[i%2](x)
		// add noise
		for j := range x.Data {
			x.Data[j] += float32(rng.NormFloat64()) * 0.02
		}
	}
	_ = rng
	opt := NewAdam(0.01, 0)
	first, last := 0.0, 0.0
	batch := 4
	for step := 0; step < 60; step++ {
		loss, correct := m.trainBatch(x, labels, batch, opt, rng)
		m.BatchStep(opt)
		if step == 0 {
			first = loss
		}
		last = loss
		_ = correct
	}
	if last > first {
		t.Fatalf("loss increased: first=%v last=%v", first, last)
	}
	// Accuracy check on the training patterns (memorization is expected here).
	probs := m.Predict(x, n)
	correct := 0
	for i := 0; i < n; i++ {
		if Argmax(probs, 2, i) == labels[i] {
			correct++
		}
	}
	if correct < 3 {
		t.Fatalf("XOR-like task not learned: correct=%d/%d loss=%v", correct, n, last)
	}
}

func fillHalf(t *Tensor, left bool) {
	_, C, H, W := t.Shape[0], t.Shape[1], t.Shape[2], t.Shape[3]
	for c := 0; c < C; c++ {
		for y := 0; y < H; y++ {
			for x := 0; x < W; x++ {
				v := float32(0)
				if (left && x < W/2) || (!left && x >= W/2) {
					v = 1
				}
				t.Data[(c*H+y)*W+x] = v
			}
		}
	}
}

// TestMetricsCorrectness validates confusion matrix and per-class metrics
// against hand-computed values.
func TestMetricsCorrectness(t *testing.T) {
	// truth [0,0,1,1,1], pred [0,1,1,1,0]
	// CM: [[1,1],[1,2]]; acc = 3/5 = 0.6
	// class0: tp=1 fp=1 fn=1 -> P=R=F1=0.5
	// class1: tp=2 fp=1 fn=1 -> P=R=F1=2/3
	truth := []int{0, 0, 1, 1, 1}
	pred := []int{0, 1, 1, 1, 0}
	cm, err := Confusion(2, truth, pred)
	if err != nil {
		t.Fatal(err)
	}
	want := [][]int{{1, 1}, {1, 2}}
	for i := range want {
		for j := range want[i] {
			if cm[i][j] != want[i][j] {
				t.Fatalf("cm[%d][%d]=%d want %d", i, j, cm[i][j], want[i][j])
			}
		}
	}
	acc, mf1, per := Evaluate(cm)
	if math.Abs(acc-0.6) > 1e-9 {
		t.Fatalf("accuracy=%v want 0.6", acc)
	}
	wantF1 := (0.5 + 2.0/3.0) / 2
	if math.Abs(mf1-wantF1) > 1e-9 {
		t.Fatalf("macroF1=%v want %v", mf1, wantF1)
	}
	if math.Abs(per[0].F1-0.5) > 1e-9 || math.Abs(per[1].F1-2.0/3.0) > 1e-9 {
		t.Fatalf("per-class F1 wrong: %+v", per)
	}
	if per[1].Support != 3 {
		t.Fatalf("support wrong: %+v", per)
	}
}

// TestModelSaveLoadRoundTrip ensures saved weights produce identical
// predictions (the basis for "save and reload the model" acceptance).
func TestModelSaveLoadRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	orig := NewCNN(16, 1, 3, 0, 1, rng)
	data, err := orig.Save("ns-cnn-small", []string{"a", "b", "c"}, true, map[string]any{"seed": 1})
	if err != nil {
		t.Fatal(err)
	}
	loaded, nm, err := LoadCNN(data)
	if err != nil {
		t.Fatal(err)
	}
	if len(nm.Classes) != 3 {
		t.Fatalf("classes=%v", nm.Classes)
	}
	x := randInput(rng, 2, 1, 16, 16)
	p1 := orig.Predict(x, 2)
	p2 := loaded.Predict(x, 2)
	for i := range p1 {
		if p1[i] != p2[i] {
			t.Fatalf("prediction mismatch at %d: %v vs %v", i, p1[i], p2[i])
		}
	}
}
