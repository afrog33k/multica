package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

const (
	defaultLocalLLMModel = "pflash-qwen3.6-27b"
	defaultZAIBaseURL    = "https://api.z.ai/api/paas/v4"
	defaultZAIModel      = "glm-4.6"
)

// openAICompatibleBackend executes prompts against an OpenAI-compatible
// /chat/completions endpoint. It is intentionally text-only: Multica still
// provides repository/workflow context, but shell/file tool execution must be
// implemented by the upstream server if needed.
type openAICompatibleBackend struct {
	cfg      Config
	provider string
}

func (b *openAICompatibleBackend) Execute(ctx context.Context, prompt string, opts ExecOptions) (*Session, error) {
	timeout := opts.Timeout
	if timeout == 0 {
		timeout = 20 * time.Minute
	}
	runCtx, cancel := context.WithTimeout(ctx, timeout)

	baseURL, apiKey, defaultModel, err := b.resolveEndpoint()
	if err != nil {
		cancel()
		return nil, err
	}
	model := strings.TrimSpace(opts.Model)
	if model == "" {
		model = defaultModel
	}
	if model == "" {
		cancel()
		return nil, fmt.Errorf("%s model is empty; set agent.model or the provider model env var", b.provider)
	}

	msgCh := make(chan Message, 16)
	resCh := make(chan Result, 1)

	go func() {
		defer cancel()
		defer close(msgCh)
		defer close(resCh)

		start := time.Now()
		trySend(msgCh, Message{Type: MessageStatus, Status: "running"})

		output, usage, err := b.chatCompletion(runCtx, baseURL, apiKey, model, prompt, opts)
		duration := time.Since(start)
		status := "completed"
		errMsg := ""
		if err != nil {
			status = "failed"
			errMsg = err.Error()
			if errors.Is(runCtx.Err(), context.DeadlineExceeded) {
				status = "timeout"
				errMsg = fmt.Sprintf("%s timed out after %s", b.provider, timeout)
			} else if errors.Is(runCtx.Err(), context.Canceled) {
				status = "aborted"
				errMsg = "execution cancelled"
			}
			trySend(msgCh, Message{Type: MessageError, Content: errMsg})
		} else if output != "" {
			trySend(msgCh, Message{Type: MessageText, Content: output})
		}

		usageMap := map[string]TokenUsage{}
		if usage.InputTokens > 0 || usage.OutputTokens > 0 || usage.CacheReadTokens > 0 || usage.CacheWriteTokens > 0 {
			usageMap[model] = usage
		}
		if len(usageMap) == 0 {
			usageMap = nil
		}

		b.cfg.Logger.Info("openai-compatible agent finished", "provider", b.provider, "status", status, "duration", duration.Round(time.Millisecond).String(), "model", model)
		resCh <- Result{
			Status:     status,
			Output:     output,
			Error:      errMsg,
			DurationMs: duration.Milliseconds(),
			Usage:      usageMap,
		}
	}()

	return &Session{Messages: msgCh, Result: resCh}, nil
}

func (b *openAICompatibleBackend) resolveEndpoint() (baseURL, apiKey, defaultModel string, err error) {
	switch b.provider {
	case "local-llm":
		baseURL = strings.TrimSpace(b.cfg.ExecutablePath)
		if baseURL == "" {
			baseURL = envFirst(b.cfg.Env, "MULTICA_LOCAL_LLM_BASE_URL", "OPENAI_BASE_URL")
		}
		apiKey = envFirst(b.cfg.Env, "MULTICA_LOCAL_LLM_API_KEY", "OPENAI_API_KEY")
		defaultModel = envFirst(b.cfg.Env, "MULTICA_LOCAL_LLM_MODEL")
		if defaultModel == "" {
			defaultModel = defaultLocalLLMModel
		}
	case "zai":
		baseURL = strings.TrimSpace(b.cfg.ExecutablePath)
		if baseURL == "" {
			baseURL = envFirst(b.cfg.Env, "MULTICA_ZAI_BASE_URL", "ZAI_BASE_URL")
		}
		if baseURL == "" {
			baseURL = defaultZAIBaseURL
		}
		apiKey = envFirst(b.cfg.Env, "MULTICA_ZAI_API_KEY", "ZAI_API_KEY")
		defaultModel = envFirst(b.cfg.Env, "MULTICA_ZAI_MODEL", "ZAI_MODEL")
		if defaultModel == "" {
			defaultModel = defaultZAIModel
		}
	default:
		return "", "", "", fmt.Errorf("unsupported OpenAI-compatible provider %q", b.provider)
	}
	if strings.TrimSpace(baseURL) == "" {
		return "", "", "", fmt.Errorf("%s base URL is empty", b.provider)
	}
	return normalizeOpenAIBaseURL(baseURL), apiKey, defaultModel, nil
}

func (b *openAICompatibleBackend) chatCompletion(ctx context.Context, baseURL, apiKey, model, prompt string, opts ExecOptions) (string, TokenUsage, error) {
	messages := make([]map[string]string, 0, 2)
	if strings.TrimSpace(opts.SystemPrompt) != "" {
		messages = append(messages, map[string]string{"role": "system", "content": opts.SystemPrompt})
	}
	messages = append(messages, map[string]string{"role": "user", "content": prompt})

	payload := map[string]any{
		"model":    model,
		"messages": messages,
		"stream":   false,
	}
	if opts.MaxTurns > 0 {
		// There is no direct OpenAI equivalent for max turns. Do not invent one.
		b.cfg.Logger.Warn("openai-compatible provider ignores max turns", "provider", b.provider, "maxTurns", opts.MaxTurns)
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return "", TokenUsage{}, fmt.Errorf("marshal request: %w", err)
	}

	endpoint := completionEndpoint(baseURL)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return "", TokenUsage{}, fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
	if b.provider == "zai" {
		req.Header.Set("User-Agent", "multica-agent/zai")
	}

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return "", TokenUsage{}, fmt.Errorf("%s chat completion request failed: %w", b.provider, err)
	}
	defer resp.Body.Close()
	respBody, readErr := io.ReadAll(io.LimitReader(resp.Body, 16*1024*1024))
	if readErr != nil {
		return "", TokenUsage{}, fmt.Errorf("read response: %w", readErr)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", TokenUsage{}, fmt.Errorf("%s chat completion returned HTTP %d: %s", b.provider, resp.StatusCode, truncateForError(string(respBody), 4096))
	}

	var decoded openAIChatResponse
	if err := json.Unmarshal(respBody, &decoded); err != nil {
		return "", TokenUsage{}, fmt.Errorf("decode response: %w", err)
	}
	if len(decoded.Choices) == 0 {
		return "", parseOpenAIUsage(decoded.Usage), fmt.Errorf("%s response had no choices", b.provider)
	}
	content := decoded.Choices[0].Message.Content
	if content == "" {
		content = decoded.Choices[0].Text
	}
	return content, parseOpenAIUsage(decoded.Usage), nil
}

func envFirst(extra map[string]string, names ...string) string {
	for _, name := range names {
		if extra != nil {
			if v, ok := extra[name]; ok && strings.TrimSpace(v) != "" {
				return strings.TrimSpace(v)
			}
		}
		if v := strings.TrimSpace(os.Getenv(name)); v != "" {
			return v
		}
	}
	return ""
}

func normalizeOpenAIBaseURL(raw string) string {
	base := strings.TrimRight(strings.TrimSpace(raw), "/")
	if strings.HasSuffix(base, "/chat/completions") {
		return strings.TrimSuffix(base, "/chat/completions")
	}
	return base
}

func completionEndpoint(baseURL string) string {
	base := strings.TrimRight(baseURL, "/")
	if strings.HasSuffix(base, "/v1") || strings.HasSuffix(base, "/v4") || strings.Contains(base, "/api/paas/v4") || strings.Contains(base, "/api/coding/paas/v4") {
		return base + "/chat/completions"
	}
	return base + "/v1/chat/completions"
}

func modelsEndpoint(baseURL string) string {
	base := strings.TrimRight(normalizeOpenAIBaseURL(baseURL), "/")
	if strings.HasSuffix(base, "/v1") || strings.HasSuffix(base, "/v4") || strings.Contains(base, "/api/paas/v4") || strings.Contains(base, "/api/coding/paas/v4") {
		return base + "/models"
	}
	return base + "/v1/models"
}

func truncateForError(s string, max int) string {
	s = strings.TrimSpace(s)
	if len(s) <= max {
		return s
	}
	return s[:max] + "..."
}

type openAIChatResponse struct {
	Choices []struct {
		Text    string `json:"text,omitempty"`
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Usage openAIUsage `json:"usage"`
}

type openAIUsage struct {
	PromptTokens             int64 `json:"prompt_tokens"`
	CompletionTokens         int64 `json:"completion_tokens"`
	InputTokens              int64 `json:"input_tokens"`
	OutputTokens             int64 `json:"output_tokens"`
	CacheReadInputTokens     int64 `json:"cache_read_input_tokens"`
	CacheCreationInputTokens int64 `json:"cache_creation_input_tokens"`
	CachedTokens             int64 `json:"cached_tokens"`
	InputTokensDetails       struct {
		CachedTokens int64 `json:"cached_tokens"`
	} `json:"input_tokens_details"`
	PromptTokensDetails struct {
		CachedTokens int64 `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
}

func parseOpenAIUsage(u openAIUsage) TokenUsage {
	input := u.PromptTokens
	if input == 0 {
		input = u.InputTokens
	}
	output := u.CompletionTokens
	if output == 0 {
		output = u.OutputTokens
	}
	cacheRead := u.CacheReadInputTokens
	if cacheRead == 0 {
		cacheRead = u.CachedTokens
	}
	if cacheRead == 0 {
		cacheRead = u.InputTokensDetails.CachedTokens
	}
	if cacheRead == 0 {
		cacheRead = u.PromptTokensDetails.CachedTokens
	}
	return TokenUsage{
		InputTokens:      input,
		OutputTokens:     output,
		CacheReadTokens:  cacheRead,
		CacheWriteTokens: u.CacheCreationInputTokens,
	}
}
