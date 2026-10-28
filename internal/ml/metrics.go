package ml

import (
	"math"
)

// Confusion builds a confusion matrix and per-class metrics from true and
// predicted label indices. All values are computed from the data — nothing
// is simulated.
func Confusion(numClasses int, truth, pred []int) ([][]int, error) {
	if len(truth) != len(pred) {
		return nil, ErrShape("truth and pred length mismatch")
	}
	cm := make([][]int, numClasses)
	for i := range cm {
		cm[i] = make([]int, numClasses)
	}
	for i := range truth {
		if truth[i] < 0 || truth[i] >= numClasses || pred[i] < 0 || pred[i] >= numClasses {
			return nil, ErrShape("label index out of range")
		}
		cm[truth[i]][pred[i]]++
	}
	return cm, nil
}

type ClassMetric struct {
	Class     string  `json:"class"`
	Precision float64 `json:"precision"`
	Recall    float64 `json:"recall"`
	F1        float64 `json:"f1"`
	Support   int     `json:"support"`
}

// Evaluate produces accuracy, macro-F1 and per-class precision/recall/F1.
func Evaluate(cm [][]int) (accuracy, macroF1 float64, perClass []ClassMetric) {
	classes := len(cm)
	total, correct := 0, 0
	perClass = make([]ClassMetric, classes)
	f1Sum := 0.0
	for c := 0; c < classes; c++ {
		tp := cm[c][c]
		fp, fn := 0, 0
		support := 0
		for r := 0; r < classes; r++ {
			for k := 0; k < classes; k++ {
				if r == c && k == c {
					continue
				}
				if k == c {
					fp += cm[r][c]
				}
				if r == c {
					fn += cm[r][k]
				}
				support += cm[r][k]
			}
		}
		// Recount support cleanly.
		support = 0
		for k := 0; k < classes; k++ {
			support += cm[c][k]
		}
		precision, recall := 0.0, 0.0
		if tp+fp > 0 {
			precision = float64(tp) / float64(tp+fp)
		}
		if tp+fn > 0 {
			recall = float64(tp) / float64(tp+fn)
		}
		f1 := 0.0
		if precision+recall > 0 {
			f1 = 2 * precision * recall / (precision + recall)
		}
		perClass[c] = ClassMetric{Precision: precision, Recall: recall, F1: f1, Support: support}
		f1Sum += f1
		for k := 0; k < classes; k++ {
			total += cm[c][k]
		}
		correct += tp
	}
	if total > 0 {
		accuracy = float64(correct) / float64(total)
	}
	if classes > 0 {
		macroF1 = f1Sum / float64(classes)
	}
	return
}

// Warnings implements honest evaluation heuristics: overfitting, imbalance.
func TrainWarnings(trainLoss, valLoss []float64, classCounts map[string]int) []string {
	var out []string
	if len(trainLoss) > 2 && len(valLoss) > 2 {
		n := len(valLoss)
		last := valLoss[n-1]
		best := last
		for _, v := range valLoss {
			if v < best {
				best = v
			}
		}
		if last > best*1.25 && last > 0 {
			out = append(out, "validation loss has risen above its best value — possible overfitting; consider early stopping or more augmentation")
		}
		if len(trainLoss) >= n && trainLoss[len(trainLoss)-1] < best*0.35 {
			out = append(out, "training loss is far below validation loss — the model may be memorizing the training split")
		}
	}
	if len(classCounts) >= 2 {
		minC, maxC := math.MaxInt64, 0
		for _, v := range classCounts {
			if v < minC {
				minC = v
			}
			if v > maxC {
				maxC = v
			}
		}
		if maxC > 0 && float64(minC)/float64(maxC) < 0.3 {
			out = append(out, "class distribution is imbalanced (smallest class has <30% of the largest) — accuracy may be misleading; check per-class metrics")
		}
	}
	return out
}

type ErrShape string

func (e ErrShape) Error() string { return string(e) }
