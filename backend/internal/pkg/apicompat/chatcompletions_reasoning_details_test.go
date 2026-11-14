//go:build unit

package apicompat

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestChatReasoningDetailsRestoreAnthropicThinkingBlocks(t *testing.T) {
	// Entries arrive out of index order and mixed with entries this bridge must ignore.
	reqJSON := `{"model":"claude-opus-5-5","messages":[
		{"role":"user","content":"look it up"},
		{"role":"assistant","content":"checking","reasoning_content":"summary must not be injected",
		 "tool_calls":[{"id":"toolu_1","type":"function","function":{"name":"lookup","arguments":"{}"}}],
		 "reasoning_details":[
			{"type":"reasoning.encrypted","data":"redacted-blob==","format":"anthropic-claude-v1","index":1},
			{"type":"reasoning.text","text":"unsigned","format":"anthropic-claude-v1","index":2},
			{"type":"reasoning.text","text":"other vendor","signature":"x","format":"openai-responses-v1","index":3},
			{"type":"reasoning.text","text":"","signature":"sig-A/B+==","format":"anthropic-claude-v1","index":0}
		 ]},
		{"role":"tool","tool_call_id":"toolu_1","content":"found"}]}`
	var req ChatCompletionsRequest
	require.NoError(t, json.Unmarshal([]byte(reqJSON), &req))

	responsesReq, err := ChatCompletionsToResponsesWithOptions(&req, ChatCompletionsToResponsesOptions{ReplayAnthropicThinking: true})
	require.NoError(t, err)
	anthropicReq, err := ResponsesToAnthropicRequest(responsesReq)
	require.NoError(t, err)

	require.Len(t, anthropicReq.Messages, 3)
	var blocks []AnthropicContentBlock
	require.NoError(t, json.Unmarshal(anthropicReq.Messages[1].Content, &blocks))
	require.Len(t, blocks, 4, "assistant content: %s", anthropicReq.Messages[1].Content)
	require.Equal(t, AnthropicContentBlock{Type: "thinking", Thinking: "", Signature: "sig-A/B+=="}, blocks[0])
	require.Equal(t, AnthropicContentBlock{Type: "redacted_thinking", Data: "redacted-blob=="}, blocks[1])
	require.Equal(t, AnthropicContentBlock{Type: "text", Text: "checking"}, blocks[2])
	require.Equal(t, "tool_use", blocks[3].Type)
	require.NotContains(t, string(anthropicReq.Messages[1].Content), "summary must not be injected")
}

func TestChatReasoningDetailsIgnoredWithoutReplay(t *testing.T) {
	reqJSON := `{"model":"claude-opus-5-5","messages":[
		{"role":"assistant","content":"checking","reasoning_content":"summary",
		 "reasoning_details":[{"type":"reasoning.encrypted","data":"redacted-blob==","format":"anthropic-claude-v1","index":0}]}]}`
	var req ChatCompletionsRequest
	require.NoError(t, json.Unmarshal([]byte(reqJSON), &req))

	responsesReq, err := ChatCompletionsToResponsesWithOptions(&req, ChatCompletionsToResponsesOptions{})
	require.NoError(t, err)

	require.NotContains(t, string(responsesReq.Input), anthropicThinkingEnvelopePrefix)
	var items []ResponsesInputItem
	require.NoError(t, json.Unmarshal(responsesReq.Input, &items))
	require.Len(t, items, 1)
	var parts []ResponsesContentPart
	require.NoError(t, json.Unmarshal(items[0].Content, &parts))
	require.Len(t, parts, 1)
	require.Equal(t, "<thinking>summary</thinking>\nchecking", parts[0].Text)
}
