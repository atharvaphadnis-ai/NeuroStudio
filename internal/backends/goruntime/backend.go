// Package goruntime is NeuroStudio's built-in training backend. It trains a
// compact convolutional network (NS-CNN-Small) for image classification
// entirely in Go on the CPU — no Python, no cloud, no external ML runtime.
// The compute is real: forward pass, backpropagation and optimizer updates
// are performed by internal/ml, which is gradient-checked by unit tests.
//
// Honest capability reporting: this backend supports exactly one task —
// image_classification at small input sizes (16–96 px). It never claims GPU
// acceleration or large-scale training; those require external backends
// delivered in later stages.
package goruntime

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"log/slog"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"

	"neurostudio/internal/backends"
	"neurostudio/internal/datasets"
	"neurostudio/internal/domain"
	"neurostudio/internal/ml"
	"neurostudio/internal/util"
)

const (
	Architecture = "ns-cnn-small"
	// maxSamples guards memory: tensors are pre-decoded into RAM.
	maxSamples = 4000
	// maxTensorBytes refuses jobs that would allocate too much image memory.
	maxTensorBytes = int64(1) << 30 // 1 GiB of float32 image tensors
)

// trainsMu serializes real training: gradient accumulation is tied to a
// package-level registry inside internal/ml, so two concurrent jobs must
// never interleave. The JobManager already enforces a single worker; this
// mutex is defense in depth.
var trainsMu sync.Mutex

type Backend struct {
	Log *slog.Logger
}

func New(log *slog.Logger) *Backend { return &Backend{Log: log} }

func (b *Backend) ID() string   { return "go-native" }
func (b *Backend) Name() string { return "NeuroStudio Native Engine (Go CPU)" }

// Discover reports real capabilities. The engine is compiled into the
// binary, so availability is a property of the build — not an assumption.
func (b *Backend) Discover(ctx context.Context) backends.Report {
	return backends.Report{
		ID:      b.ID(),
		Name:    b.Name(),
		Status:  "available",
		Version: "1.0.0",
		Tasks: map[string]backends.Compatibility{
			string(domain.TaskImageClassification): {Supported: true},
			string(domain.TaskObjectDetection):     {Supported: false, Reason: "not implemented by the native engine"},
			string(domain.TaskAudioClassification): {Supported: false, Reason: "not implemented by the native engine"},
			string(domain.TaskTextClassification):  {Supported: false, Reason: "not implemented by the native engine"},
		},
		Accelerators: []string{"cpu"},
		Detail: map[string]string{
			"compute":     "pure Go conv/dense kernels, numerically gradient-checked",
			"inputRange":  "16–96 px square images (RGB or grayscale)",
			"scaling":     "designed for laptop-scale datasets (tens to a few thousand images)",
			"parallelism": "convolution parallelized across CPU cores",
		},
		CheckedAt: time.Now().UTC(),
	}
}

// Train runs one complete image-classification job: load manifest, decode
// and preprocess images, train with real backprop, checkpoint, early-stop,
// evaluate the held-out split and serialize the best model.
func (b *Backend) Train(ctx context.Context, spec backends.TrainSpec, emit func(domain.JobEvent)) (*backends.TrainResult, error) {
	trainsMu.Lock()
	defer trainsMu.Unlock()

	started := time.Now()
	cfg := spec.Config

	// ---- normalize configuration honestly ----
	cfg.Input.Size = clampi(cfg.Input.Size, 16, 96)
	cfg.Epochs = clampi(cfg.Epochs, 1, 500)
	cfg.BatchSize = clampi(cfg.BatchSize, 1, 128)
	if cfg.LearningRate <= 0 {
		cfg.LearningRate = 1e-3
	}
	cfg.Workers = clampi(cfg.Workers, 1, 16)
	if cfg.Seed == 0 {
		cfg.Seed = 42
	}
	if cfg.Optimizer != "sgd" && cfg.Optimizer != "adam" {
		cfg.Optimizer = "adam"
	}
	rng := rand.New(rand.NewSource(cfg.Seed))

	// ---- load dataset manifest ----
	emit(phase("prepare", "loading dataset manifest"))
	raw, err := os.ReadFile(spec.ManifestPath)
	if err != nil {
		return nil, fmt.Errorf("read manifest: %w", err)
	}
	var manifest datasets.Manifest
	if err := json.Unmarshal(raw, &manifest); err != nil {
		return nil, fmt.Errorf("parse manifest: %w", err)
	}
	if manifest.Format != "neurostudio/manifest-1" {
		return nil, fmt.Errorf("unsupported manifest format %q", manifest.Format)
	}
	classes := manifest.Classes
	if len(classes) < 2 {
		return nil, util.ErrBadRequest("training requires at least 2 classes (found %d)", len(classes))
	}
	classIndex := map[string]int{}
	for i, c := range classes {
		classIndex[c] = i
	}

	// ---- decode + preprocess usable samples ----
	emit(phase("prepare", "decoding and preprocessing images"))
	var trainIdx, valIdx, testIdx []int
	var tensors []*ml.Tensor
	labels := []int{}
	var totalBytes int64
	perSample := int64(cfg.Input.Size) * int64(cfg.Input.Size) * int64(cfg.Channels()) * 4
	decodeFailed := 0

	for _, sm := range manifest.Samples {
		if ctx.Err() != nil {
			return nil, util.ErrCancelled
		}
		if len(tensors) >= maxSamples {
			emit(domain.JobEvent{Type: domain.EvtLog, Payload: map[string]string{
				"message": fmt.Sprintf("native engine loaded the first %d samples (memory guard); larger datasets need an external backend", maxSamples),
			}})
			break
		}
		if totalBytes+perSample > maxTensorBytes {
			emit(domain.JobEvent{Type: domain.EvtLog, Payload: map[string]string{
				"message": fmt.Sprintf("stopping data load at %d samples to stay within the 1 GB memory guard", len(tensors)),
			}})
			break
		}
		if sm.W > 8192 || sm.H > 8192 {
			decodeFailed++
			continue
		}
		path := filepath.Join(spec.DataRoot, manifest.DatasetID, filepath.FromSlash(sm.Path))
		data, err := os.ReadFile(path)
		if err != nil {
			decodeFailed++
			continue
		}
		img, _, err := image.Decode(bytes.NewReader(data))
		if err != nil {
			decodeFailed++
			continue
		}
		t, err := ml.ImageToTensor(img, cfg.Input.Size, cfg.Input.Grayscale)
		if err != nil {
			decodeFailed++
			continue
		}
		ci, ok := classIndex[sm.Class]
		if !ok {
			decodeFailed++
			continue
		}
		tensors = append(tensors, t)
		labels = append(labels, ci)
		totalBytes += perSample
		switch sm.Split {
		case domain.SplitTrain:
			trainIdx = append(trainIdx, len(tensors)-1)
		case domain.SplitVal:
			valIdx = append(valIdx, len(tensors)-1)
		case domain.SplitTest:
			testIdx = append(testIdx, len(tensors)-1)
		}
	}
	if decodeFailed > 0 {
		emit(domain.JobEvent{Type: domain.EvtLog, Payload: map[string]string{
			"message": fmt.Sprintf("%d samples could not be decoded and were skipped", decodeFailed),
		}})
	}
	if len(trainIdx) == 0 {
		return nil, util.ErrBadRequest("no usable training samples after preprocessing")
	}
	if len(valIdx) == 0 && len(testIdx) == 0 {
		return nil, util.ErrBadRequest("no held-out samples: create a dataset version with a val or test split before training")
	}
	if len(valIdx) == 0 {
		valIdx, testIdx = testIdx, nil
		emit(domain.JobEvent{Type: domain.EvtLog, Payload: map[string]string{
			"message": "no validation split found — the test split is being used for validation; final metrics report this",
		}})
	}

	classCounts := map[string]int{}
	for _, li := range labels {
		classCounts[classes[li]]++
	}
	emit(domain.JobEvent{Type: domain.EvtLog, Payload: map[string]string{
		"message": fmt.Sprintf("dataset ready: %d train / %d val / %d test, %d classes, input %dx%d %s",
			len(trainIdx), len(valIdx), len(testIdx), len(classes), cfg.Input.Size, cfg.Input.Size, chName(cfg.Channels())),
	}})

	// ---- model + optimizer ----
	emit(phase("train", "initializing model"))
	model := ml.NewCNN(cfg.Input.Size, cfg.Channels(), len(classes), int(cfg.Dropout*10), cfg.Workers, rng)
	var opt ml.Optimizer
	if cfg.Optimizer == "sgd" {
		opt = ml.NewSGD(cfg.LearningRate, 0.9, cfg.WeightDecay)
	} else {
		opt = ml.NewAdam(cfg.LearningRate, cfg.WeightDecay)
	}

	stepsPerEpoch := (len(trainIdx) + cfg.BatchSize - 1) / cfg.BatchSize
	type bestState struct {
		epoch  int
		loss   float64
		params []*ml.Tensor
	}
	var best bestState
	var curves []domain.CurvePt
	var trainLossHist, valLossHist []float64
	noImprove := 0
	patience := cfg.EarlyStopping.Patience
	if patience < 1 {
		patience = 5
	}

	saveCheckpoint := func(epoch int, valLoss, valAcc float64) (string, error) {
		ckptDir := filepath.Join(spec.WorkDir, "checkpoints")
		if err := os.MkdirAll(ckptDir, 0o755); err != nil {
			return "", err
		}
		data, err := model.Save(Architecture, classes, cfg.Input.Grayscale, map[string]any{
			"epoch": epoch, "jobId": spec.JobID, "valLoss": valLoss, "valAcc": valAcc,
		})
		if err != nil {
			return "", err
		}
		p := filepath.Join(ckptDir, fmt.Sprintf("epoch-%03d.json", epoch))
		return p, util.AtomicWrite(p, data, 0o644)
	}

	// ---- training loop (real forward + backward + optimizer step) ----
	for epoch := 0; epoch < cfg.Epochs; epoch++ {
		if ctx.Err() != nil {
			return nil, util.ErrCancelled
		}
		lr := ml.ScheduleLR(cfg.LearningRate, cfg.LRSchedule, epoch, cfg.Epochs)
		setLR(opt, lr)

		order := append([]int(nil), trainIdx...)
		rng.Shuffle(len(order), func(i, j int) { order[i], order[j] = order[j], order[i] })

		var epochLoss float64
		var epochCorrect int
		batch := ml.NewTensor(cfg.BatchSize, cfg.Channels(), cfg.Input.Size, cfg.Input.Size)

		for s := 0; s < stepsPerEpoch; s++ {
			if ctx.Err() != nil {
				return nil, util.ErrCancelled
			}
			lo, hi := s*cfg.BatchSize, (s+1)*cfg.BatchSize
			if hi > len(order) {
				hi = len(order)
			}
			n := hi - lo
			lbls := make([]int, n)
			for bi := 0; bi < n; bi++ {
				src := tensors[order[lo+bi]]
				var t *ml.Tensor
				if AugmentationEnabled(cfg) {
					t = ml.Augment(src, cfg.Augmentation, rng) // train split only
				} else {
					t = src
				}
				copy(batch.Data[bi*src.Size():(bi+1)*src.Size()], t.Data)
				lbls[bi] = labels[order[lo+bi]]
			}
			loss, correct := model.TrainBatch(batch, lbls, n, opt, rng)
			model.BatchStep(opt) // apply accumulated gradients (ml contract)
			epochLoss += loss
			epochCorrect += correct

			emit(domain.JobEvent{Type: "tick", Payload: map[string]any{
				"epoch": epoch + 1, "epochs": cfg.Epochs, "step": s + 1, "steps": stepsPerEpoch,
				"trainLoss": round4(epochLoss / float64(s+1)),
			}})
		}
		trainLoss := epochLoss / float64(stepsPerEpoch)

		// Validation on held-out data, no augmentation, no dropout.
		valLoss, valAcc, _ := evaluate(model, tensors, labels, valIdx, cfg.Input)

		curves = append(curves, domain.CurvePt{
			Epoch: epoch + 1, TrainLoss: round4(trainLoss), ValLoss: round4(valLoss),
			ValAcc: round4(valAcc), LR: round4(lr),
		})
		trainLossHist = append(trainLossHist, trainLoss)
		valLossHist = append(valLossHist, valLoss)

		emit(domain.JobEvent{Type: domain.EvtMetrics, Payload: domain.EpochMetrics{
			Epoch: epoch + 1, Epochs: cfg.Epochs, Step: stepsPerEpoch, Steps: stepsPerEpoch,
			TrainLoss: round4(trainLoss), ValLoss: round4(valLoss), ValAcc: round4(valAcc), LR: round4(lr),
		}})

		if best.params == nil || valLoss < best.loss {
			best = bestState{epoch: epoch + 1, loss: valLoss, params: cloneParams(model)}
			noImprove = 0
		} else {
			noImprove++
		}

		if cfg.CheckpointEveryEpochs > 0 && (epoch+1)%cfg.CheckpointEveryEpochs == 0 {
			if p, err := saveCheckpoint(epoch+1, valLoss, valAcc); err == nil {
				emit(domain.JobEvent{Type: domain.EvtCheckpoint, Payload: map[string]any{
					"epoch": epoch + 1, "path": p, "valLoss": round4(valLoss), "valAcc": round4(valAcc),
				}})
			}
		}

		if cfg.EarlyStopping.Enabled && noImprove >= patience {
			emit(domain.JobEvent{Type: domain.EvtLog, Payload: map[string]string{
				"message": fmt.Sprintf("early stopping: validation loss has not improved for %d epochs", patience),
			}})
			break
		}
	}

	// ---- restore best-validation weights ----
	if best.params != nil {
		restoreParams(model, best.params)
	}

	// ---- held-out evaluation (never used for training) ----
	emit(phase("evaluate", "evaluating on the held-out split"))
	evalSplit, evalName := testIdx, "test"
	if len(evalSplit) == 0 {
		evalSplit, evalName = valIdx, "val"
	}
	testLoss, testAcc, cm := evaluate(model, tensors, labels, evalSplit, cfg.Input)
	_, macroF1, perClass := ml.Evaluate(cm)
	warnings := ml.TrainWarnings(trainLossHist, valLossHist, classCounts)
	if evalName == "val" {
		warnings = append(warnings, "final metrics were computed on the validation split because no test split existed")
	}
	latP50, latP95 := latency(model, tensors, evalSplit, cfg.Input)

	// ---- save the model ----
	emit(phase("done", "saving model"))
	modelPath := filepath.Join(spec.WorkDir, "model.native.json")
	data, err := model.Save(Architecture, classes, cfg.Input.Grayscale, map[string]any{
		"jobId": spec.JobID, "trainedWith": "neurostudio go-native engine",
		"bestEpoch": best.epoch, "seed": cfg.Seed,
	})
	if err != nil {
		return nil, fmt.Errorf("serialize model: %w", err)
	}
	if err := util.AtomicWrite(modelPath, data, 0o644); err != nil {
		return nil, fmt.Errorf("write model: %w", err)
	}

	perClassAny := make([]map[string]any, len(perClass))
	for i, pc := range perClass {
		perClassAny[i] = map[string]any{
			"class": classes[i], "precision": round4(pc.Precision),
			"recall": round4(pc.Recall), "f1": round4(pc.F1), "support": pc.Support,
		}
	}

	metrics := map[string]any{
		"bestEpoch":        best.epoch,
		"bestValLoss":      round4(best.loss),
		"finalTrainLoss":   round4(curves[len(curves)-1].TrainLoss),
		"evalSplit":        evalName,
		"evalAccuracy":     round4(testAcc),
		"evalLoss":         round4(testLoss),
		"evalMacroF1":      round4(macroF1),
		"confusion":        cm,
		"classes":          classes,
		"perClass":         perClassAny,
		"latencyP50Ms":     round4(latP50),
		"latencyP95Ms":     round4(latP95),
		"trainSamples":     len(trainIdx),
		"valSamples":       len(valIdx),
		"testSamples":      len(testIdx),
		"trainTimeSeconds": round4(time.Since(started).Seconds()),
		"warnings":         warnings,
		"architecture":     Architecture,
		"input":            cfg.Input,
	}

	return &backends.TrainResult{
		ModelPath:    modelPath,
		Format:       "native-1",
		Architecture: Architecture,
		Classes:      classes,
		Input:        cfg.Input,
		Metrics:      metrics,
		Curves:       curves,
		LogPath:      filepath.Join(spec.WorkDir, "training.log"),
	}, nil
}

// evaluate runs inference over the given sample indices and returns mean
// loss, accuracy and the confusion matrix. Nothing is simulated.
func evaluate(model *ml.CNN, tensors []*ml.Tensor, labels []int, idxs []int, in domain.InputSpec) (loss, acc float64, cm [][]int) {
	if len(idxs) == 0 {
		return 0, 0, nil
	}
	ch := 1
	if !in.Grayscale {
		ch = 3
	}
	const batch = 32
	correct, total := 0, 0
	totalLoss := 0.0
	cm = make([][]int, model.Classes)
	for i := range cm {
		cm[i] = make([]int, model.Classes)
	}
	for lo := 0; lo < len(idxs); lo += batch {
		hi := lo + batch
		if hi > len(idxs) {
			hi = len(idxs)
		}
		n := hi - lo
		sub := ml.NewTensor(n, ch, model.InputSize, model.InputSize)
		lbls := make([]int, n)
		for bi := 0; bi < n; bi++ {
			src := tensors[idxs[lo+bi]]
			copy(sub.Data[bi*src.Size():(bi+1)*src.Size()], src.Data)
			lbls[bi] = labels[idxs[lo+bi]]
		}
		logits := model.ForwardLogits(sub, n)
		l, _, probs := ml.SoftmaxCE(logits, lbls, n, model.Classes)
		totalLoss += l * float64(n)
		for bi := 0; bi < n; bi++ {
			pred := ml.Argmax(probs, model.Classes, bi)
			cm[lbls[bi]][pred]++
			if pred == lbls[bi] {
				correct++
			}
		}
		total += n
	}
	if total == 0 {
		return 0, 0, cm
	}
	return totalLoss / float64(total), float64(correct) / float64(total), cm
}

// latency measures real per-image inference time over the eval split.
func latency(model *ml.CNN, tensors []*ml.Tensor, idxs []int, in domain.InputSpec) (p50, p95 float64) {
	if len(idxs) == 0 {
		return 0, 0
	}
	ch := 1
	if !in.Grayscale {
		ch = 3
	}
	const batch = 32
	var durs []float64
	for lo := 0; lo < len(idxs); lo += batch {
		hi := lo + batch
		if hi > len(idxs) {
			hi = len(idxs)
		}
		n := hi - lo
		sub := ml.NewTensor(n, ch, model.InputSize, model.InputSize)
		for bi := 0; bi < n; bi++ {
			src := tensors[idxs[lo+bi]]
			copy(sub.Data[bi*src.Size():(bi+1)*src.Size()], src.Data)
		}
		t0 := time.Now()
		_ = model.Predict(sub, n)
		durs = append(durs, float64(time.Since(t0).Microseconds())/1000/float64(n))
	}
	if len(durs) == 0 {
		return 0, 0
	}
	sort.Float64s(durs)
	return durs[len(durs)/2], durs[int(float64(len(durs)-1)*0.95)]
}

func cloneParams(m *ml.CNN) []*ml.Tensor {
	out := make([]*ml.Tensor, 0, 8)
	for _, p := range m.ParamTensors() {
		out = append(out, p.Clone())
	}
	return out
}

func restoreParams(m *ml.CNN, saved []*ml.Tensor) {
	for i, p := range m.ParamTensors() {
		copy(p.Data, saved[i].Data)
	}
}

func setLR(opt ml.Optimizer, lr float64) {
	switch o := opt.(type) {
	case *ml.Adam:
		o.LearningRate = lr
	case *ml.SGD:
		o.LearningRate = lr
	}
}

// AugmentationEnabled reports whether any augmentation is configured.
// Augmentation applies to the training split only — val/test are untouched.
func AugmentationEnabled(cfg domain.TrainConfig) bool {
	a := cfg.Augmentation
	return a.HFlip || a.Rotate || a.Brightness || a.Contrast || a.Noise || a.CropPad > 0
}

func phase(name, msg string) domain.JobEvent {
	return domain.JobEvent{Type: domain.EvtPhase, Payload: map[string]string{"phase": name, "message": msg}}
}

func round4(v float64) float64 { return math.Round(v*10000) / 10000 }

func clampi(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func chName(ch int) string {
	if ch == 1 {
		return "grayscale"
	}
	return "RGB"
}
