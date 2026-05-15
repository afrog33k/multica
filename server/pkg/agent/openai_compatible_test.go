package agent

import (
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestOpenAICompatibleBackendExecute(t *testing.T) {
	t.Parallel()

	var gotModel string
	var gotSystem string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("path = %q", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Fatalf("Authorization = %q", r.Header.Get("Authorization"))
		}
		var req struct {
			Model    string `json:"model"`
			Messages []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		gotModel = req.Model
		if len(req.Messages) < 2 {
			t.Fatalf("expected system and user messages, got %+v", req.Messages)
		}
		gotSystem = req.Messages[0].Content
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]any{"content": "done"}}},
			"usage":   map[string]any{"prompt_tokens": 7, "completion_tokens": 3, "input_tokens_details": map[string]any{"cached_tokens": 2}},
		})
	}))
	defer srv.Close()

	backend := &openAICompatibleBackend{provider: "local-llm", cfg: Config{
		ExecutablePath: srv.URL,
		Logger:         slog.Default(),
		Env: map[string]string{
			"MULTICA_LOCAL_LLM_API_KEY": "test-key",
			"MULTICA_LOCAL_LLM_MODEL":   "local-model",
		},
	}}
	session, err := backend.Execute(context.Background(), "ship it", ExecOptions{SystemPrompt: "system", Timeout: time.Second})
	if err != nil {
		t.Fatalf("Execute error: %v", err)
	}
	res := <-session.Result
	if res.Status != "completed" || res.Output != "done" {
		t.Fatalf("result = %+v", res)
	}
	if gotModel != "local-model" || gotSystem != "system" {
		t.Fatalf("request model/system = %q/%q", gotModel, gotSystem)
	}
	usage := res.Usage["local-model"]
	if usage.InputTokens != 7 || usage.OutputTokens != 3 || usage.CacheReadTokens != 2 {
		t.Fatalf("usage = %+v", usage)
	}
}

func TestOpenAICompatibleEndpointHelpers(t *testing.T) {
	t.Parallel()
	if got := completionEndpoint("http://host:8100/v1"); got != "http://host:8100/v1/chat/completions" {
		t.Fatalf("completionEndpoint v1 = %q", got)
	}
	if got := completionEndpoint("https://api.z.ai/api/paas/v4"); got != "https://api.z.ai/api/paas/v4/chat/completions" {
		t.Fatalf("completionEndpoint zai = %q", got)
	}
	if got := modelsEndpoint("http://host:8100"); got != "http://host:8100/v1/models" {
		t.Fatalf("modelsEndpoint root = %q", got)
	}
}
