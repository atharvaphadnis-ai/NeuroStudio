package ml

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand"
)

// Architecture: NS-CNN-Small — a compact convolutional network designed for
// CPU training on small images (<= 64px). Three conv blocks followed by a
// dense classifier head. Parameters: ~26k + head.

type CNN struct {
	InputSize int
	Channels  int
	Classes   int
	Conv1     *Conv2D
	Conv2     *Conv2D
	Conv3     *Conv2D
	Dense     *Dense
	Dropout   float64
	workers   int
}

func NewCNN(inputSize, channels, classes, dropout int, workers int, rng *rand.Rand) *CNN {
	if workers < 1 {
		workers = 1
	}
	if inputSize < 16 || inputSize > 96 {
		inputSize = 48
	}
	return &CNN{
		InputSize: inputSize, Channels: channels, Classes: classes,
		Conv1:   NewConv2D(16, channels, 3, 1, 1, rng),
		Conv2:   NewConv2D(32, 16, 3, 1, 1, rng),
		Conv3:   NewConv2D(64, 32, 3, 1, 1, rng),
		Dense:   NewDense(64*(inputSize/8)*(inputSize/8), classes, rng),
		Dropout: float64(dropout),
		workers: workers,
	}
}

// forward returns logits for a batch [n, C, H, W]. If wantProbs, also
// returns softmax probabilities.
func (m *CNN) forward(x *Tensor, n int, train bool, rng *rand.Rand) (logits *Tensor, probs []float32, a *acts) {
	a = &acts{}
	a.x1 = x
	z1 := m.Conv1.Forward(x, n, m.workers)
	a.z1 = z1
	h1 := ReLU(z1)
	a.h1 = h1
	p1, a1 := MaxPool2{}.Forward(h1, n)
	a.p1, a.a1 = p1, a1

	z2 := m.Conv2.Forward(p1, n, m.workers)
	a.z2 = z2
	h2 := ReLU(z2)
	a.h2 = h2
	p2, a2 := MaxPool2{}.Forward(h2, n)
	a.p2, a.a2 = p2, a2

	z3 := m.Conv3.Forward(p2, n, m.workers)
	a.z3 = z3
	h3 := ReLU(z3)
	a.h3 = h3
	p3, a3 := MaxPool2{}.Forward(h3, n)
	a.p3, a.a3 = p3, a3

	a.flat = FlattenBatch(p3, n)
	fc := m.Dense.Forward(a.flat, n)

	if train && m.Dropout > 0 && m.Dropout < 0.9 {
		keep := float32(1 - m.Dropout)
		scale := 1 / keep
		for i := range fc.Data {
			if rng.Float64() < m.Dropout {
				fc.Data[i] = 0
			} else {
				fc.Data[i] *= scale
			}
		}
	}

	_, _, probs = SoftmaxCE(fc, make([]int, n), n, m.Classes)
	return fc, probs, a
}

type acts struct {
	x1, z1, h1, p1, z2, h2, p2, z3, h3, p3, flat *Tensor
	a1, a2, a3                                   []MaxArg
}

// trainBatch runs forward + backward for one batch and applies the update.
func (m *CNN) trainBatch(x *Tensor, labels []int, n int, opt Optimizer, rng *rand.Rand) (loss float64, correct int) {
	logits, probs, a := m.forward(x, n, true, rng)
	loss, dLogits, _ := SoftmaxCE(logits, labels, n, m.Classes)
	for ni := 0; ni < n; ni++ {
		if Argmax(probs, m.Classes, ni) == labels[ni] {
			correct++
		}
	}

	// Dense head backward.
	dFlat := NewTensor(a.flat.Shape...)
	{
		dW, dB, dIn := m.Dense.Backward(a.flat, dLogits, n)
		// apply with zero-grad accumulation below
		m.applyGrads(opt, []*Tensor{m.Dense.W, m.Dense.B}, []*Tensor{dW, dB})
		copy(dFlat.Data, dIn.Data)
	}
	// Undo dropout mask on gradients is not needed (inverted dropout).

	dP3 := &Tensor{Shape: a.p3.Shape, Data: dFlat.Data}
	dH3 := MaxPool2Backward(a.h3, dP3, a.a3, n)
	dZ3 := ReLUBackward(a.h3, dH3)
	{
		dW := NewTensor(m.Conv3.W.Shape...)
		dB := NewTensor(m.Conv3.B.Shape...)
		dP2 := m.Conv3.Backward(a.p2, dZ3, n, dW, dB, m.workers)
		m.applyGrads(opt, []*Tensor{m.Conv3.W, m.Conv3.B}, []*Tensor{dW, dB})

		dH2 := MaxPool2Backward(a.h2, dP2, a.a2, n)
		dZ2 := ReLUBackward(a.h2, dH2)
		dW2 := NewTensor(m.Conv2.W.Shape...)
		dB2 := NewTensor(m.Conv2.B.Shape...)
		dP1 := m.Conv2.Backward(a.p1, dZ2, n, dW2, dB2, m.workers)
		m.applyGrads(opt, []*Tensor{m.Conv2.W, m.Conv2.B}, []*Tensor{dW2, dB2})

		dH1 := MaxPool2Backward(a.h1, dP1, a.a1, n)
		dZ1 := ReLUBackward(a.h1, dH1)
		dW1 := NewTensor(m.Conv1.W.Shape...)
		dB1 := NewTensor(m.Conv1.B.Shape...)
		_ = m.Conv1.Backward(a.x1, dZ1, n, dW1, dB1, m.workers)
		m.applyGrads(opt, []*Tensor{m.Conv1.W, m.Conv1.B}, []*Tensor{dW1, dB1})
	}
	return loss, correct
}

var gradBuf = map[*Tensor][]*Tensor{}

// applyGrads collects per-parameter gradients; optimizers step once per
// batch, so we store them until BatchStep is invoked by the trainer.
func (m *CNN) applyGrads(opt Optimizer, params, grads []*Tensor) {
	for i, p := range params {
		gradBuf[p] = append(gradBuf[p], grads[i])
	}
}

// BatchStep applies the accumulated gradients and clears them.
func (m *CNN) BatchStep(opt Optimizer) {
	seen := map[*Tensor]bool{}
	var params []*Tensor
	var total []*Tensor
	for p, gs := range gradBuf {
		if seen[p] {
			continue
		}
		seen[p] = true
		sum := NewTensor(p.Shape...)
		for _, g := range gs {
			for i := range sum.Data {
				sum.Data[i] += g.Data[i]
			}
		}
		params = append(params, p)
		total = append(total, sum)
	}
	opt.Step(params, total)
	for p := range gradBuf {
		delete(gradBuf, p)
	}
}

// Predict runs inference on a batch and returns probabilities.
func (m *CNN) Predict(x *Tensor, n int) []float32 {
	_, probs, _ := m.forward(x, n, false, nil)
	return probs
}

// ForwardLogits exposes the forward pass for trainers that need raw logits
// (e.g. to compute evaluation loss with their own label slice).
func (m *CNN) ForwardLogits(x *Tensor, n int) *Tensor {
	logits, _, _ := m.forward(x, n, false, nil)
	return logits
}

// TrainBatch performs one real optimization step on a batch:
// forward pass, backward pass, gradient application. Returns (loss, correct).
func (m *CNN) TrainBatch(x *Tensor, labels []int, n int, opt Optimizer, rng *rand.Rand) (float64, int) {
	return m.trainBatch(x, labels, n, opt, rng)
}

// ParamTensors returns all trainable tensors in a stable order.
func (m *CNN) ParamTensors() []*Tensor {
	return []*Tensor{m.Conv1.W, m.Conv1.B, m.Conv2.W, m.Conv2.B, m.Conv3.W, m.Conv3.B, m.Dense.W, m.Dense.B}
}

// ---- serialization (native model format) ----

type NativeModel struct {
	Format       string `json:"format"` // "neurostudio/native-1"
	Architecture string `json:"architecture"`
	Task         string `json:"task"`
	Input        struct {
		Size      int     `json:"size"`
		Grayscale bool    `json:"grayscale"`
		Scale     float64 `json:"scale"`
	} `json:"input"`
	Classes []string          `json:"classes"`
	Dropout float64           `json:"dropout"`
	Weights map[string]string `json:"weights"` // name -> base64 float32 LE
	Meta    map[string]any    `json:"meta"`
}

const nativeFormat = "neurostudio/native-1"

func b64(f32 []float32) string {
	buf := make([]byte, 4*len(f32))
	for i, v := range f32 {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	return base64.StdEncoding.EncodeToString(buf)
}

func unb64(s string) ([]float32, error) {
	raw, err := base64.StdEncoding.DecodeString(s)
	if err != nil {
		return nil, err
	}
	n := len(raw) / 4
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out, nil
}

// Save serializes the model to the native JSON format.
func (m *CNN) Save(arch string, classes []string, grayscale bool, meta map[string]any) ([]byte, error) {
	nm := NativeModel{Format: nativeFormat, Architecture: arch, Task: "image_classification",
		Classes: classes, Weights: map[string]string{}, Meta: meta}
	nm.Input.Size = m.InputSize
	nm.Input.Grayscale = grayscale
	nm.Input.Scale = 1.0 / 255.0
	nm.Dropout = m.Dropout
	nm.Weights["conv1_w"] = b64(m.Conv1.W.Data)
	nm.Weights["conv1_b"] = b64(m.Conv1.B.Data)
	nm.Weights["conv2_w"] = b64(m.Conv2.W.Data)
	nm.Weights["conv2_b"] = b64(m.Conv2.B.Data)
	nm.Weights["conv3_w"] = b64(m.Conv3.W.Data)
	nm.Weights["conv3_b"] = b64(m.Conv3.B.Data)
	nm.Weights["dense_w"] = b64(m.Dense.W.Data)
	nm.Weights["dense_b"] = b64(m.Dense.B.Data)
	return json.MarshalIndent(nm, "", " ")
}

// LoadCNN reconstructs a CNN from a native model JSON.
func LoadCNN(data []byte) (*CNN, *NativeModel, error) {
	var nm NativeModel
	if err := json.Unmarshal(data, &nm); err != nil {
		return nil, nil, fmt.Errorf("invalid model file: %w", err)
	}
	if nm.Format != nativeFormat {
		return nil, nil, fmt.Errorf("unsupported model format %q (expected %s)", nm.Format, nativeFormat)
	}
	if nm.Architecture != "ns-cnn-small" {
		return nil, nil, fmt.Errorf("unsupported architecture %q", nm.Architecture)
	}
	if len(nm.Classes) == 0 {
		return nil, nil, errors.New("model has no classes")
	}
	classes := len(nm.Classes)
	rng := rand.New(rand.NewSource(1))
	m := NewCNN(nm.Input.Size, 3, classes, 0, 1, rng)
	if nm.Input.Grayscale {
		m.Channels = 1
		m.Conv1 = NewConv2D(16, 1, 3, 1, 1, rng)
	}
	var err error
	if m.Conv1.W.Data, err = unb64(nm.Weights["conv1_w"]); err != nil {
		return nil, nil, fmt.Errorf("conv1_w: %w", err)
	}
	if m.Conv1.B.Data, err = unb64(nm.Weights["conv1_b"]); err != nil {
		return nil, nil, fmt.Errorf("conv1_b: %w", err)
	}
	if m.Conv2.W.Data, err = unb64(nm.Weights["conv2_w"]); err != nil {
		return nil, nil, fmt.Errorf("conv2_w: %w", err)
	}
	if m.Conv2.B.Data, err = unb64(nm.Weights["conv2_b"]); err != nil {
		return nil, nil, fmt.Errorf("conv2_b: %w", err)
	}
	if m.Conv3.W.Data, err = unb64(nm.Weights["conv3_w"]); err != nil {
		return nil, nil, fmt.Errorf("conv3_w: %w", err)
	}
	if m.Conv3.B.Data, err = unb64(nm.Weights["conv3_b"]); err != nil {
		return nil, nil, fmt.Errorf("conv3_b: %w", err)
	}
	if m.Dense.W.Data, err = unb64(nm.Weights["dense_w"]); err != nil {
		return nil, nil, fmt.Errorf("dense_w: %w", err)
	}
	if m.Dense.B.Data, err = unb64(nm.Weights["dense_b"]); err != nil {
		return nil, nil, fmt.Errorf("dense_b: %w", err)
	}
	// Validate shapes.
	if len(m.Dense.W.Data) != m.Dense.Out*m.Dense.In {
		return nil, nil, errors.New("dense weight shape mismatch")
	}
	return m, &nm, nil
}
