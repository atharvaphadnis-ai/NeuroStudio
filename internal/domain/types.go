// Package domain defines the shared vocabulary of NeuroStudio: projects,
// datasets, training jobs, experiments, models and the events that flow
// between the API, the job manager and the training backends.
package domain

import "time"

// TaskType identifies a machine-learning task. A task may only be advertised
// as supported when a real preprocessing + training pipeline exists for it.
type TaskType string

const (
	TaskImageClassification TaskType = "image_classification"
	TaskObjectDetection     TaskType = "object_detection"
	TaskImageSegmentation   TaskType = "image_segmentation"
	TaskAudioClassification TaskType = "audio_classification"
	TaskPoseClassification  TaskType = "pose_classification"
	TaskTextClassification  TaskType = "text_classification"
	TaskTabular             TaskType = "tabular"
	TaskTimeSeries          TaskType = "time_series"
)

// DataType identifies the kind of media a dataset contains.
type DataType string

const (
	DataImage    DataType = "image"
	DataAudio    DataType = "audio"
	DataVideo    DataType = "video"
	DataTabular  DataType = "tabular"
	DataTypeText DataType = "text"
	DataSeries   DataType = "time_series"
)

// Job lifecycle states. Interrupted jobs are marked failed at startup with a
// clear reason; NeuroStudio never silently re-runs expensive training.
type JobStatus string

const (
	JobQueued    JobStatus = "queued"
	JobRunning   JobStatus = "running"
	JobCompleted JobStatus = "completed"
	JobFailed    JobStatus = "failed"
	JobCancelled JobStatus = "cancelled"
)

// Sample statuses produced by dataset validation.
const (
	SampleOK        = "ok"
	SampleCorrupt   = "corrupt"
	SampleDuplicate = "duplicate"
)

// Split labels.
const (
	SplitTrain = "train"
	SplitVal   = "val"
	SplitTest  = "test"
	SplitNone  = "" // not assigned yet
)

type Project struct {
	ID          string         `json:"id"`
	Name        string         `json:"name"`
	Description string         `json:"description"`
	TaskType    TaskType       `json:"taskType"`
	Status      string         `json:"status"` // active | archived
	Settings    map[string]any `json:"settings"`
	CreatedAt   time.Time      `json:"createdAt"`
	UpdatedAt   time.Time      `json:"updatedAt"`
	ArchivedAt  *time.Time     `json:"archivedAt,omitempty"`
	Counts      ProjectCounts  `json:"counts,omitempty"`
}

type ProjectCounts struct {
	Datasets    int `json:"datasets"`
	Samples     int `json:"samples"`
	Experiments int `json:"experiments"`
	Models      int `json:"models"`
	JobsRunning int `json:"jobsRunning"`
}

type Dataset struct {
	ID        string    `json:"id"`
	ProjectID string    `json:"projectId"`
	Name      string    `json:"name"`
	DataType  DataType  `json:"dataType"`
	Notes     string    `json:"notes"`
	CreatedAt time.Time `json:"createdAt"`
	UpdatedAt time.Time `json:"updatedAt"`

	// Populated on read.
	ClassCount  int           `json:"classCount"`
	SampleCount int           `json:"sampleCount"`
	Stats       *DatasetStats `json:"stats,omitempty"`
	Classes     []Class       `json:"classes,omitempty"`
}

type Class struct {
	ID        string    `json:"id"`
	DatasetID string    `json:"datasetId"`
	Name      string    `json:"name"`
	Color     string    `json:"color"`
	Ord       int       `json:"ord"`
	Samples   int       `json:"samples,omitempty"`
	CreatedAt time.Time `json:"createdAt"`
}

type Sample struct {
	ID         string         `json:"id"`
	DatasetID  string         `json:"datasetId"`
	ClassID    *string        `json:"classId,omitempty"`
	Class      string         `json:"class,omitempty"`
	Filename   string         `json:"filename"`
	RelPath    string         `json:"relPath"`
	MediaType  string         `json:"mediaType"`
	SizeBytes  int64          `json:"sizeBytes"`
	SHA256     string         `json:"sha256"`
	Width      int            `json:"width,omitempty"`
	Height     int            `json:"height,omitempty"`
	DurationMS int            `json:"durationMs,omitempty"`
	Status     string         `json:"status"`
	Meta       map[string]any `json:"meta"`
	CreatedAt  time.Time      `json:"createdAt"`
}

type DatasetStats struct {
	TotalSamples int            `json:"totalSamples"`
	ByStatus     map[string]int `json:"byStatus"`
	ByClass      map[string]int `json:"byClass"`
	TotalBytes   int64          `json:"totalBytes"`
	AvgBytes     int64          `json:"avgBytes"`
	Widths       MinMax         `json:"widths"`
	Heights      MinMax         `json:"heights"`
}

type MinMax struct {
	Min int `json:"min"`
	Max int `json:"max"`
}

type DatasetVersion struct {
	ID           string              `json:"id"`
	DatasetID    string              `json:"datasetId"`
	Version      int                 `json:"version"`
	Seed         int64               `json:"seed"`
	Params       SplitParams         `json:"params"`
	Stats        DatasetVersionStats `json:"stats"`
	ManifestPath string              `json:"-"`
	CreatedAt    time.Time           `json:"createdAt"`
	// split assignment sampleID -> train|val|test (loaded from splits JSON)
	Splits map[string]string `json:"-"`
}

type SplitParams struct {
	Train float64 `json:"train"`
	Val   float64 `json:"val"`
	Test  float64 `json:"test"`
	Seed  int64   `json:"seed"`
}

type DatasetVersionStats struct {
	Counts     map[string]int            `json:"counts"`   // split -> n
	ByClass    map[string]map[string]int `json:"byClass"`  // split -> class -> n
	Warnings   []string                  `json:"warnings"` // leakage, imbalance
	TotalBytes int64                     `json:"totalBytes"`
}

type Annotation struct {
	ID        string         `json:"id"`
	SampleID  string         `json:"sampleId"`
	Kind      string         `json:"kind"` // bbox | label
	Label     string         `json:"label"`
	Data      map[string]any `json:"data"` // bbox: {x,y,w,h} normalized 0..1
	CreatedAt time.Time      `json:"createdAt"`
	UpdatedAt time.Time      `json:"updatedAt"`
}

// ---- Training ----

type TrainConfig struct {
	TaskType     TaskType `json:"taskType"`
	Architecture string   `json:"architecture"`

	// Input pipeline.
	Input InputSpec `json:"input"`

	// Optimization.
	Epochs       int     `json:"epochs"`
	BatchSize    int     `json:"batchSize"`
	LearningRate float64 `json:"learningRate"`
	Optimizer    string  `json:"optimizer"` // adam | sgd
	WeightDecay  float64 `json:"weightDecay"`
	LRSchedule   string  `json:"lrSchedule"` // none | cosine | step
	Seed         int64   `json:"seed"`
	Workers      int     `json:"workers"`

	// Regularization / stopping.
	EarlyStopping struct {
		Enabled  bool    `json:"enabled"`
		Patience int     `json:"patience"`
		MinDelta float64 `json:"minDelta"`
	} `json:"earlyStopping"`
	Dropout float64 `json:"dropout"`

	// Augmentation applied to the training split only.
	Augmentation AugmentConfig `json:"augmentation"`

	// Checkpointing.
	CheckpointEveryEpochs int `json:"checkpointEveryEpochs"`

	// Split policy used when the dataset has no explicit version.
	SplitPolicy *SplitParams `json:"splitPolicy,omitempty"`

	// Fine-tuning (used by backends that support pretrained weights).
	FineTune struct {
		Pretrained  string `json:"pretrained,omitempty"` // weight set id
		Strategy    string `json:"strategy,omitempty"`   // freeze | partial | full
		TrainableTo int    `json:"trainableTo,omitempty"`
	} `json:"fineTune"`
}

type InputSpec struct {
	Size      int  `json:"size"`      // square edge in pixels
	Grayscale bool `json:"grayscale"` // 1 channel instead of 3
}

func (c TrainConfig) Channels() int {
	if c.Input.Grayscale {
		return 1
	}
	return 3
}

type AugmentConfig struct {
	HFlip      bool    `json:"hFlip"`
	Rotate     bool    `json:"rotate"`
	MaxAngle   float64 `json:"maxAngle"`
	Brightness bool    `json:"brightness"`
	Contrast   bool    `json:"contrast"`
	Noise      bool    `json:"noise"`
	CropPad    int     `json:"cropPad"`
}

// Job is a persistent training job record.
type Job struct {
	ID               string      `json:"id"`
	ProjectID        string      `json:"projectId"`
	DatasetID        string      `json:"datasetId,omitempty"`
	DatasetVersionID string      `json:"datasetVersionId,omitempty"`
	Type             string      `json:"type"` // train
	BackendID        string      `json:"backendId"`
	Status           JobStatus   `json:"status"`
	Name             string      `json:"name"`
	Config           TrainConfig `json:"config"`
	Progress         Progress    `json:"progress"`
	Error            string      `json:"error,omitempty"`
	CancelRequested  bool        `json:"cancelRequested"`
	CreatedAt        time.Time   `json:"createdAt"`
	StartedAt        *time.Time  `json:"startedAt,omitempty"`
	FinishedAt       *time.Time  `json:"finishedAt,omitempty"`
}

type Progress struct {
	Phase         string    `json:"phase"` // prepare | train | evaluate | done
	Epoch         int       `json:"epoch"`
	Epochs        int       `json:"epochs"`
	Step          int       `json:"step"`
	StepsPerEpoch int       `json:"stepsPerEpoch"`
	TrainLoss     float64   `json:"trainLoss"`
	ValLoss       float64   `json:"valLoss"`
	ValAccuracy   float64   `json:"valAccuracy"`
	BestValLoss   float64   `json:"bestValLoss"`
	BestEpoch     int       `json:"bestEpoch"`
	LR            float64   `json:"lr"`
	ElapsedMS     int64     `json:"elapsedMs"`
	ETASeconds    float64   `json:"etaSeconds"`
	Curves        []CurvePt `json:"curves,omitempty"`
}

type CurvePt struct {
	Epoch     int     `json:"epoch"`
	TrainLoss float64 `json:"trainLoss"`
	ValLoss   float64 `json:"valLoss"`
	ValAcc    float64 `json:"valAcc"`
	LR        float64 `json:"lr"`
}

// JobEvent is streamed to SSE clients and persisted for replay.
type JobEvent struct {
	JobID   string    `json:"jobId"`
	Seq     int64     `json:"seq"`
	Type    string    `json:"type"`
	Time    time.Time `json:"time"`
	Payload any       `json:"payload"`
}

const (
	EvtLog        = "log"
	EvtMetrics    = "metrics"
	EvtPhase      = "phase"
	EvtCheckpoint = "checkpoint"
	EvtSystem     = "system"
	EvtDone       = "done"
	EvtError      = "error"
	EvtStatus     = "status"
)

type EpochMetrics struct {
	Epoch     int     `json:"epoch"`
	Epochs    int     `json:"epochs"`
	Step      int     `json:"step"`
	Steps     int     `json:"steps"`
	TrainLoss float64 `json:"trainLoss"`
	ValLoss   float64 `json:"valLoss"`
	ValAcc    float64 `json:"valAcc"`
	LR        float64 `json:"lr"`
}

type Checkpoint struct {
	ID        string             `json:"id"`
	JobID     string             `json:"jobId"`
	Epoch     int                `json:"epoch"`
	Path      string             `json:"-"`
	Metrics   map[string]float64 `json:"metrics"`
	CreatedAt time.Time          `json:"createdAt"`
}

// ---- Experiments & models ----

type Experiment struct {
	ID               string         `json:"id"`
	ProjectID        string         `json:"projectId"`
	JobID            string         `json:"jobId,omitempty"`
	Name             string         `json:"name"`
	TaskType         TaskType       `json:"taskType"`
	BackendID        string         `json:"backendId"`
	BackendVersion   string         `json:"backendVersion"`
	Architecture     string         `json:"architecture"`
	Seed             int64          `json:"seed"`
	Config           TrainConfig    `json:"config"`
	DatasetID        string         `json:"datasetId,omitempty"`
	DatasetVersionID string         `json:"datasetVersionId,omitempty"`
	Metrics          map[string]any `json:"metrics"`
	Curves           []CurvePt      `json:"curves"`
	Artifacts        []ArtifactRef  `json:"artifacts"`
	Runtime          map[string]any `json:"runtime"`
	Notes            string         `json:"notes"`
	CreatedAt        time.Time      `json:"createdAt"`
}

type ArtifactRef struct {
	Kind string `json:"kind"` // model | checkpoint | export | log
	Path string `json:"path"`
	Size int64  `json:"size"`
	Name string `json:"name"`
}

type Evaluation struct {
	Split        string         `json:"split"`
	Accuracy     float64        `json:"accuracy"`
	MacroF1      float64        `json:"macroF1"`
	Confusion    [][]int        `json:"confusion"`
	Classes      []string       `json:"classes"`
	PerClass     []ClassMetrics `json:"perClass"`
	LatencyP50MS float64        `json:"latencyP50Ms"`
	LatencyP95MS float64        `json:"latencyP95Ms"`
	NumSamples   int            `json:"numSamples"`
	Warnings     []string       `json:"warnings"`
}

type ClassMetrics struct {
	Class     string  `json:"class"`
	Precision float64 `json:"precision"`
	Recall    float64 `json:"recall"`
	F1        float64 `json:"f1"`
	Support   int     `json:"support"`
}

type Model struct {
	ID           string         `json:"id"`
	ProjectID    string         `json:"projectId"`
	Name         string         `json:"name"`
	TaskType     TaskType       `json:"taskType"`
	Architecture string         `json:"architecture"`
	Notes        string         `json:"notes"`
	CreatedAt    time.Time      `json:"createdAt"`
	UpdatedAt    time.Time      `json:"updatedAt"`
	Versions     []ModelVersion `json:"versions,omitempty"`
}

type ModelVersion struct {
	ID            string         `json:"id"`
	ModelID       string         `json:"modelId"`
	Version       int            `json:"version"`
	ExperimentID  string         `json:"experimentId,omitempty"`
	Format        string         `json:"format"` // native-1 | onnx
	FilePath      string         `json:"-"`
	FileSize      int64          `json:"fileSize"`
	SHA256        string         `json:"sha256"`
	InputSpec     InputSpec      `json:"inputSpec"`
	OutputSpec    map[string]any `json:"outputSpec"`
	Classes       []string       `json:"classes"`
	Metrics       map[string]any `json:"metrics"`
	ExportFormats []string       `json:"exportFormats"`
	License       string         `json:"license"`
	RuntimeReqs   map[string]any `json:"runtimeReqs"`
	Notes         string         `json:"notes"`
	CreatedAt     time.Time      `json:"createdAt"`
}

type Prediction struct {
	Label string  `json:"label"`
	Score float64 `json:"score"`
}

type InferenceResult struct {
	Predictions    []Prediction `json:"predictions"`
	LatencyMS      float64      `json:"latencyMs"`
	Runtime        string       `json:"runtime"` // go-native | browser
	ModelVersionID string       `json:"modelVersionId"`
}
