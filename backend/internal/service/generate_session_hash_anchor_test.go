//go:build unit

package service

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func mustParseChatCompletionsSessionHashRequest(t *testing.T, body string, ctx *SessionContext) *ParsedRequest {
	t.Helper()
	parsed, err := ParseGatewayRequest(NewRequestBodyRef([]byte(body)), "chat_completions")
	require.NoError(t, err)
	parsed.SessionContext = ctx
	return parsed
}

// A client without metadata.user_id keeps one sticky key for the whole
// conversation, so every round lands on the account that holds its prompt
// cache. Breakpoints placed on the latest messages move each round and must
// not decide the key.
func TestGenerateSessionHash_ConversationAnchorStaysStableAcrossRounds(t *testing.T) {
	svc := &GatewayService{}
	ctx := &SessionContext{ClientIP: "127.0.0.1", UserAgent: "test-client", APIKeyID: 11}
	marked := func(text string) map[string]any {
		return msg("user", []any{map[string]any{"type": "text", "text": text, "cache_control": map[string]any{"type": "ephemeral"}}})
	}
	rounds := [][]any{
		{marked("summarize order 42")},
		{msg("user", "summarize order 42"), msg("assistant", "calling tool"), marked("tool results 1")},
		{msg("user", "summarize order 42"), msg("assistant", "calling tool"), marked("tool results 1"), msg("assistant", "calling tool again"), marked("tool results 2")},
	}
	var hashes []string
	for _, messages := range rounds {
		hashes = append(hashes, svc.GenerateSessionHash(mustParseSessionHashRequest(t, anthropicSessionBody("You are a helpful assistant.", messages, ""), ctx)))
	}
	require.NotEmpty(t, hashes[0])
	require.Equal(t, hashes[0], hashes[1])
	require.Equal(t, hashes[1], hashes[2])

	other := svc.GenerateSessionHash(mustParseSessionHashRequest(t, anthropicSessionBody("You are a helpful assistant.", []any{marked("summarize order 43")}, ""), ctx))
	require.NotEqual(t, hashes[0], other, "a conversation with another first message is another session")

	geminiCtx := &SessionContext{ClientIP: "10.0.0.1", UserAgent: "gemini-cli", APIKeyID: 42}
	round1 := svc.GenerateSessionHash(mustParseGeminiSessionHashRequest(t, geminiSessionBody(nil, []any{geminiMsg("user", "Write a Go function")}), geminiCtx))
	round2 := svc.GenerateSessionHash(mustParseGeminiSessionHashRequest(t, geminiSessionBody(nil, []any{geminiMsg("user", "Write a Go function"), geminiMsg("model", "func hello() {}"), geminiMsg("user", "Add tests")}), geminiCtx))
	require.NotEmpty(t, round1)
	require.Equal(t, round1, round2)
}

// On the chat completions shape the system prompt is messages[0]. The anchor
// is the leading system messages plus the first other message; the system
// prompt alone must not be the anchor, otherwise every conversation of one
// client shares one key.
func TestGenerateSessionHash_ChatCompletionsAnchorIncludesSystemAndFirstMessage(t *testing.T) {
	svc := &GatewayService{}
	ctx := &SessionContext{ClientIP: "127.0.0.1", UserAgent: "test-client", APIKeyID: 11}
	hash := func(body string) string {
		return svc.GenerateSessionHash(mustParseChatCompletionsSessionHashRequest(t, body, ctx))
	}

	round1 := hash(`{"model":"my-model","messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"summarize order 42"}]}`)
	round2 := hash(`{"model":"my-model","messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"summarize order 42"},{"role":"assistant","content":"order 42 shipped"},{"role":"user","content":"and order 43?"}]}`)
	require.NotEmpty(t, round1)
	require.Equal(t, round1, round2, "rounds of one conversation share a key")

	other := hash(`{"model":"my-model","messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"draft a release note"}]}`)
	require.NotEqual(t, round1, other, "same system prompt with another first user message is another conversation")

	otherSystem := hash(`{"model":"my-model","messages":[{"role":"system","content":"You are a terse assistant."},{"role":"user","content":"summarize order 42"}]}`)
	require.NotEqual(t, round1, otherSystem, "same first user message with another system message is another conversation")

	withDeveloper := hash(`{"model":"my-model","messages":[{"role":"developer","content":"Answer briefly."},{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"summarize order 42"}]}`)
	require.NotEqual(t, round1, withDeveloper, "a leading developer message is part of the anchor")
}

// A breakpoint on messages moves with every round, so such a request skips the
// cacheable content key and uses the conversation anchor. A breakpoint on the
// system prompt alone keeps its content key.
func TestGenerateSessionHash_MessageBreakpointSkipsCacheableContentKey(t *testing.T) {
	svc := &GatewayService{}
	ctxA := &SessionContext{ClientIP: "10.0.0.1", UserAgent: "test-client", APIKeyID: 1}
	ctxB := &SessionContext{ClientIP: "10.0.0.2", UserAgent: "test-client", APIKeyID: 2}
	cc := map[string]any{"type": "ephemeral"}
	system := []any{map[string]any{"type": "text", "text": "You are a helpful assistant.", "cache_control": cc}}
	marked := msg("user", []any{map[string]any{"type": "text", "text": "summarize order 42", "cache_control": cc}})

	withMessageBreakpoint := anthropicSessionBody(system, []any{marked}, "")
	a := svc.GenerateSessionHash(mustParseSessionHashRequest(t, withMessageBreakpoint, ctxA))
	b := svc.GenerateSessionHash(mustParseSessionHashRequest(t, withMessageBreakpoint, ctxB))
	require.NotEqual(t, a, b, "a message breakpoint skips the content key, so the request context is part of the key")

	systemOnly := func(first string, ctx *SessionContext) string {
		return svc.GenerateSessionHash(mustParseSessionHashRequest(t, anthropicSessionBody(system, []any{msg("user", first)}, ""), ctx))
	}
	require.Equal(t, systemOnly("summarize order 42", ctxA), systemOnly("draft a release note", ctxB), "a system-only breakpoint keeps its content key")
}
