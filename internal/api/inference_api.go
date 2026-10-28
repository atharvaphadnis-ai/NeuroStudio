package api

import (
	"bytes"
	"database/sql"
	"encoding/json"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"net/http"
	"os"
	"time"

	"neurostudio/internal/domain"
	"neurostudio/internal/ml"
	"neurostudio/internal/util"
)

// registerInferenceRoutes exposes local inference against registered native
// models. Replaces the Stage 1 stub. Predictions are computed by the same
// gradient-checked kernels used for training — nothing is simulated.
func (s *Server) registerInferenceRoutes(m *http.ServeMux) {
	m.HandleFunc("POST /api/v1/models/{id}/versions/{vid}/predict", s.handlePredict)
	m.HandleFunc("POST /api/v1/models/{id}/predict", s.handlePredictLatest)
}

// handlePredictLatest predicts with the newest version of the model.
func (s *Server) handlePredictLatest(w http.ResponseWriter, r *http.Request) {
	var vid string
	err := s.Store.DB().QueryRowContext(r.Context(),
		`SELECT id FROM model_versions WHERE model_id=? ORDER BY version DESC LIMIT 1`,
		r.PathValue("id")).Scan(&vid)
	if err == sql.ErrNoRows {
		util.WriteErr(w, util.ErrNotFound("model version"))
		return
	}
	if err != nil {
		util.WriteErr(w, err)
		return
	}
	s.predictWith(w, r, r.PathValue("id"), vid)
}

func (s *Server) handlePredict(w http.ResponseWriter, r *http.Request) {
	s.predictWith(w, r, r.PathValue("id"), r.PathValue("vid"))
}

func (s *Server) predictWith(w http.ResponseWriter, r *http.Request, modelID, versionID string) {
	// Resolve the version and verify it belongs to the model.
	var path, format, classesJSON, inputJSON string
	err := s.Store.DB().QueryRowContext(r.Context(),
		`SELECT file_path, format, classes, input_spec FROM model_versions WHERE id=? AND model_id=?`,
		versionID, modelID).Scan(&path, &format, &classesJSON, &inputJSON)
	if err == sql.ErrNoRows {
		util.WriteErr(w, util.ErrNotFound("model version"))
		return
	}
	if err != nil {
		util.WriteErr(w, err)
		return
	}
	if format != "native-1" {
		util.WriteErr(w, util.Err(422, "unsupported_format",
			fmt.Sprintf("inference for format %q is not supported by this runtime", format)))
		return
	}
	data, err := os.ReadFile(path)
	if err != nil {
		util.WriteErr(w, util.Err(410, "artifact_missing", "model file is no longer on disk"))
		return
	}
	model, nm, err := ml.LoadCNN(data)
	if err != nil {
		util.WriteErr(w, util.Err(422, "invalid_model", err.Error()))
		return
	}

	// Read the uploaded image (bounded).
	r.Body = http.MaxBytesReader(w, r.Body, 32<<20)
	if err := r.ParseMultipartForm(32 << 20); err != nil {
		util.WriteErr(w, util.ErrBadRequest("multipart form with an 'image' field is required"))
		return
	}
	file, _, err := r.FormFile("image")
	if err != nil {
		util.WriteErr(w, util.ErrBadRequest("missing 'image' file field"))
		return
	}
	defer file.Close()
	imgData := bytes.Buffer{}
	if _, err := imgData.ReadFrom(file); err != nil {
		util.WriteErr(w, util.ErrBadRequest("cannot read image"))
		return
	}
	img, _, err := image.Decode(bytes.NewReader(imgData.Bytes()))
	if err != nil {
		util.WriteErr(w, util.Err(422, "bad_image", "uploaded file is not a decodable PNG/JPEG/GIF image"))
		return
	}

	// Preprocess exactly like training (same size, same scale).
	tensor, err := ml.ImageToTensor(img, model.InputSize, nm.Input.Grayscale)
	if err != nil {
		util.WriteErr(w, util.Err(422, "bad_image", err.Error()))
		return
	}

	t0 := time.Now()
	probs := model.Predict(tensor, 1)
	latency := float64(time.Since(t0).Microseconds()) / 1000

	var classes []string
	_ = json.Unmarshal([]byte(classesJSON), &classes)
	out := make([]domain.Prediction, len(classes))
	for i := range classes {
		out[i] = domain.Prediction{Label: classes[i], Score: round6(float64(probs[i]))}
	}
	// Sort descending by score.
	for i := 0; i < len(out); i++ {
		for j := i + 1; j < len(out); j++ {
			if out[j].Score > out[i].Score {
				out[i], out[j] = out[j], out[i]
			}
		}
	}
	util.WriteJSON(w, 200, domain.InferenceResult{
		Predictions: out, LatencyMS: round6(latency),
		Runtime: "go-native", ModelVersionID: versionID,
	})
}

func round6(v float64) float64 {
	return float64(int(v*1e6+0.5)) / 1e6
}
