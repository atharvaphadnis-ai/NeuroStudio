package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"neurostudio/internal/domain"
	"neurostudio/internal/training"
	"neurostudio/internal/util"
)

// registerTrainingRoutes wires the training endpoints. Replaces the Stage 1
// stub: these routes only exist because the implementation behind them does.
func (s *Server) registerTrainingRoutes(m *http.ServeMux) {
	m.HandleFunc("GET /api/v1/training/backends", s.handleBackends)
	m.HandleFunc("POST /api/v1/projects/{id}/jobs", s.handleJobCreate)
	m.HandleFunc("GET /api/v1/projects/{id}/jobs", s.handleJobList)
	m.HandleFunc("GET /api/v1/jobs/{id}", s.handleJobGet)
	m.HandleFunc("POST /api/v1/jobs/{id}/cancel", s.handleJobCancel)
	m.HandleFunc("GET /api/v1/jobs/{id}/events", s.handleJobEvents)
	m.HandleFunc("GET /api/v1/jobs/{id}/checkpoints", s.handleJobCheckpoints)
	m.HandleFunc("GET /api/v1/jobs/{id}/evaluation", s.handleJobEvaluation)
}

func (s *Server) handleBackends(w http.ResponseWriter, r *http.Request) {
	reports := s.Trainer.Registry().DiscoverAll(r.Context())
	util.WriteJSON(w, 200, map[string]any{"backends": reports})
}

func (s *Server) handleJobCreate(w http.ResponseWriter, r *http.Request) {
	var in training.CreateInput
	if err := util.DecodeJSON(r, &in); err != nil {
		util.WriteErr(w, err)
		return
	}
	job, err := s.Trainer.Enqueue(r.Context(), r.PathValue("id"), in)
	if err != nil {
		util.WriteErr(w, err)
		return
	}
	util.WriteJSON(w, 201, job)
}

func (s *Server) handleJobList(w http.ResponseWriter, r *http.Request) {
	jobs, err := s.Trainer.ListJobs(r.Context(), r.PathValue("id"), 100)
	if err != nil {
		util.WriteErr(w, err)
		return
	}
	util.WriteJSON(w, 200, jobs)
}

func (s *Server) handleJobGet(w http.ResponseWriter, r *http.Request) {
	job, err := s.Trainer.GetJob(r.Context(), r.PathValue("id"))
	if err != nil {
		util.WriteErr(w, err)
		return
	}
	util.WriteJSON(w, 200, job)
}

func (s *Server) handleJobCancel(w http.ResponseWriter, r *http.Request) {
	job, err := s.Trainer.Cancel(r.Context(), r.PathValue("id"))
	if err != nil {
		util.WriteErr(w, err)
		return
	}
	util.WriteJSON(w, 200, job)
}

func (s *Server) handleJobCheckpoints(w http.ResponseWriter, r *http.Request) {
	jobID := r.PathValue("id")
	rows, err := s.Store.DB().QueryContext(r.Context(),
		`SELECT id, epoch, metrics, created_at FROM checkpoints WHERE job_id=? ORDER BY epoch`, jobID)
	if err != nil {
		util.WriteErr(w, err)
		return
	}
	defer rows.Close()
	type ckpt struct {
		ID        string         `json:"id"`
		Epoch     int            `json:"epoch"`
		Metrics   map[string]any `json:"metrics"`
		CreatedAt string         `json:"createdAt"`
	}
	out := []ckpt{}
	for rows.Next() {
		var c ckpt
		var metrics string
		if err := rows.Scan(&c.ID, &c.Epoch, &metrics, &c.CreatedAt); err != nil {
			util.WriteErr(w, err)
			return
		}
		_ = json.Unmarshal([]byte(metrics), &c.Metrics)
		out = append(out, c)
	}
	util.WriteJSON(w, 200, out)
}

// handleJobEvaluation returns the final evaluation of the job's experiment
// (test-split accuracy, macro-F1, confusion matrix, per-class metrics).
func (s *Server) handleJobEvaluation(w http.ResponseWriter, r *http.Request) {
	jobID := r.PathValue("id")
	var metrics, curves string
	var expID string
	err := s.Store.DB().QueryRowContext(r.Context(),
		`SELECT id, metrics, curves FROM experiments WHERE job_id=? ORDER BY created_at DESC LIMIT 1`, jobID).
		Scan(&expID, &metrics, &curves)
	if err != nil {
		util.WriteErr(w, util.Err(409, "not_ready",
			"no evaluation available yet — the job has not completed successfully"))
		return
	}
	var mj any
	_ = json.Unmarshal([]byte(metrics), &mj)
	var cj []domain.CurvePt
	_ = json.Unmarshal([]byte(curves), &cj)
	util.WriteJSON(w, 200, map[string]any{"experimentId": expID, "metrics": mj, "curves": cj})
}

// handleJobEvents streams the job event log over SSE. Persisted events are
// replayed first (supporting Last-Event-ID resumption), then live events are
// forwarded until the job reaches a terminal state or the client leaves.
func (s *Server) handleJobEvents(w http.ResponseWriter, r *http.Request) {
	jobID := r.PathValue("id")
	job, err := s.Trainer.GetJob(r.Context(), jobID)
	if err != nil {
		util.WriteErr(w, err)
		return
	}
	fl, ok := w.(http.Flusher)
	if !ok {
		util.WriteErr(w, util.Err(500, "no_stream", "streaming unsupported"))
		return
	}
	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("Connection", "keep-alive")
	h.Set("X-Accel-Buffering", "no")

	after := int64(0)
	if v := r.Header.Get("Last-Event-ID"); v != "" {
		fmt.Sscanf(v, "%d", &after)
	}
	if v := r.URL.Query().Get("after"); v != "" {
		fmt.Sscanf(v, "%d", &after)
	}

	// Subscribe before replay so no event falls in the gap; dedupe by seq.
	live, off := s.Trainer.Hub().Subscribe(jobID)
	defer off()

	writeEvent := func(evt domain.JobEvent) {
		if evt.Seq > 0 {
			fmt.Fprintf(w, "id: %d\n", evt.Seq)
		}
		fmt.Fprintf(w, "event: %s\n", evt.Type)
		data, _ := json.Marshal(evt)
		fmt.Fprintf(w, "data: %s\n\n", data)
		fl.Flush()
	}

	// Replay persisted history.
	history, err := s.Trainer.Events(jobID, after)
	if err != nil {
		return
	}
	lastSent := after
	for _, evt := range history {
		writeEvent(evt)
		if evt.Seq > lastSent {
			lastSent = evt.Seq
		}
	}

	// If the job is already terminal and history covers it, stop after replay.
	terminal := job.Status == domain.JobCompleted || job.Status == domain.JobFailed ||
		job.Status == domain.JobCancelled
	if terminal {
		// Ensure a done event was part of the replay; if not, synthesize one.
		sawDone := false
		for _, evt := range history {
			if evt.Type == domain.EvtDone {
				sawDone = true
			}
		}
		if sawDone {
			fmt.Fprint(w, "retry: 5000\n\n")
			fl.Flush()
			return
		}
	}

	heartbeat := time.NewTicker(15 * time.Second)
	defer heartbeat.Stop()
	for {
		select {
		case <-r.Context().Done():
			return
		case <-heartbeat.C:
			fmt.Fprint(w, ": ping\n\n")
			fl.Flush()
		case evt := <-live:
			if evt.Seq > 0 && evt.Seq <= lastSent {
				continue // already replayed
			}
			if evt.Seq > lastSent {
				lastSent = evt.Seq
			}
			writeEvent(evt)
			if evt.Type == domain.EvtDone {
				return
			}
		}
	}
}
