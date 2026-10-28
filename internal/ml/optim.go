package ml

import (
	"math"
	"math/rand"
)

// Optimizer applies parameter updates for one trainable tensor.
type Optimizer interface {
	Step(params, grads []*Tensor)
	LR() float64
}

// SGD with momentum.
type SGD struct {
	LearningRate, Momentum, WeightDecay float64
	vel                                 map[*Tensor]*Tensor
}

func NewSGD(lr, momentum, wd float64) *SGD {
	return &SGD{LearningRate: lr, Momentum: momentum, WeightDecay: wd, vel: map[*Tensor]*Tensor{}}
}

func (o *SGD) LR() float64 { return o.LearningRate }

func (o *SGD) Step(params, grads []*Tensor) {
	for i, p := range params {
		g := grads[i]
		v, ok := o.vel[p]
		if !ok {
			v = NewTensor(p.Shape...)
			o.vel[p] = v
		}
		for j := range p.Data {
			gv := g.Data[j]
			if o.WeightDecay > 0 {
				gv += float32(o.WeightDecay) * p.Data[j]
			}
			v.Data[j] = float32(o.Momentum)*v.Data[j] + gv
			p.Data[j] -= float32(o.LearningRate) * v.Data[j]
		}
	}
}

// Adam optimizer with decoupled state (m, v, step count).
type Adam struct {
	LearningRate, Beta1, Beta2, Eps, WeightDecay float64
	t                                            int
	m, v                                         map[*Tensor]*Tensor
}

func NewAdam(lr, wd float64) *Adam {
	return &Adam{LearningRate: lr, Beta1: 0.9, Beta2: 0.999, Eps: 1e-8,
		WeightDecay: wd, m: map[*Tensor]*Tensor{}, v: map[*Tensor]*Tensor{}}
}

func (o *Adam) LR() float64 { return o.LearningRate }

func (o *Adam) Step(params, grads []*Tensor) {
	o.t++
	b1, b2 := o.Beta1, o.Beta2
	bc1 := 1 - math.Pow(b1, float64(o.t))
	bc2 := 1 - math.Pow(b2, float64(o.t))
	for i, p := range params {
		g := grads[i]
		m, ok := o.m[p]
		if !ok {
			m = NewTensor(p.Shape...)
			o.m[p] = m
			o.v[p] = NewTensor(p.Shape...)
		}
		v := o.v[p]
		for j := range p.Data {
			gv := float64(g.Data[j])
			if o.WeightDecay > 0 {
				gv += o.WeightDecay * float64(p.Data[j])
			}
			m.Data[j] = float32(b1*float64(m.Data[j]) + (1-b1)*gv)
			v.Data[j] = float32(b2*float64(v.Data[j]) + (1-b2)*gv*gv)
			mh := float64(m.Data[j]) / bc1
			vh := float64(v.Data[j]) / bc2
			p.Data[j] -= float32(o.LearningRate * mh / (math.Sqrt(vh) + o.Eps))
		}
	}
}

// Learning-rate schedules.
func ScheduleLR(base float64, kind string, epoch, epochs int) float64 {
	switch kind {
	case "cosine":
		if epochs <= 1 {
			return base
		}
		return base * 0.5 * (1 + math.Cos(math.Pi*float64(epoch)/float64(epochs-1)))
	case "step":
		if epoch >= 10 && epoch%10 == 0 {
			return base * math.Pow(0.5, float64(epoch/10))
		}
		return base
	default:
		return base
	}
}

var _ = rand.Int
