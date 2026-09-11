package responses

import (
	"strings"

	translatorcommon "github.com/router-for-me/CLIProxyAPI/v7/internal/translator/common"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// ConvertOpenAIResponsesRequestToOpenAIChatCompletions converts OpenAI responses format to OpenAI chat completions format.
// It transforms the OpenAI responses API format (with instructions and input array) into the standard
// OpenAI chat completions format (with messages array and system content).
//
// The conversion handles:
// 1. Model name and streaming configuration
// 2. Instructions to system message conversion
// 3. Input array to messages array transformation
// 4. Tool definitions and tool choice conversion
// 5. Function calls and function results handling
// 6. Generation parameters mapping (max_tokens, reasoning, etc.)
//
// Parameters:
//   - modelName: The name of the model to use for the request
//   - rawJSON: The raw JSON request data in OpenAI responses format
//   - stream: A boolean indicating if the request is for a streaming response
//
// Returns:
//   - []byte: The transformed request data in OpenAI chat completions format
func ConvertOpenAIResponsesRequestToOpenAIChatCompletions(modelName string, inputRawJSON []byte, stream bool) []byte {
	rawJSON := inputRawJSON
	// Base OpenAI chat completions template with default values
	out := []byte(`{"model":"","messages":[],"stream":false}`)

	root := gjson.ParseBytes(rawJSON)
	toolIndex := newResponsesToolIndex(root)

	messages := make([][]byte, 0)
	// OpenAI tool messages cannot carry image parts, so images returned by a
	// tool output are relayed as a user message following the tool message run.
	relayedToolImages := make([][]byte, 0)
	flushRelayImages := func() {
		if len(relayedToolImages) == 0 {
			return
		}
		relayItems := make([][]byte, 0, len(relayedToolImages)+1)
		noticeJSON := []byte(`{"type":"text","text":""}`)
		noticeJSON, _ = sjson.SetBytes(noticeJSON, "text", toolResultImageRelayNotice)
		relayItems = append(relayItems, noticeJSON)
		relayItems = append(relayItems, relayedToolImages...)
		relayJSON := []byte(`{"role":"user"}`)
		relayJSON, _ = sjson.SetRawBytes(relayJSON, "content", translatorcommon.JoinRawArray(relayItems))
		messages = append(messages, relayJSON)
		relayedToolImages = relayedToolImages[:0]
	}
	appendMessage := func(message []byte) {
		switch gjson.GetBytes(message, "role").String() {
		case "tool":
			// Tool messages stay adjacent so tool-call responses keep their group.
		case "user":
			// Merge relayed images into the following user message so the request
			// keeps a single user turn.
			if len(relayedToolImages) > 0 {
				message = mergeRelayImagesIntoUserMessage(message, relayedToolImages)
				relayedToolImages = relayedToolImages[:0]
			}
		default:
			flushRelayImages()
		}
		messages = append(messages, message)
	}

	// Set model name
	out, _ = sjson.SetBytes(out, "model", modelName)

	// Set stream configuration
	out, _ = sjson.SetBytes(out, "stream", stream)

	// Map Responses text format to Chat Completions response format.
	if textFormat := root.Get("text.format"); textFormat.Exists() {
		if responseFormat := convertResponsesTextFormatToChatResponseFormat(textFormat); len(responseFormat) > 0 {
			out, _ = sjson.SetRawBytes(out, "response_format", responseFormat)
		}
	}

	// Map generation parameters from responses format to chat completions format
	if maxTokens := root.Get("max_output_tokens"); maxTokens.Exists() {
		if maxTokens.Raw != "" {
			out, _ = sjson.SetRawBytes(out, "max_tokens", []byte(maxTokens.Raw))
		} else {
			out, _ = sjson.SetBytes(out, "max_tokens", maxTokens.Value())
		}
	}

	// Convert instructions to system message
	if instructions := root.Get("instructions"); instructions.Exists() {
		systemMessage := []byte(`{"role":"system","content":""}`)
		systemMessage, _ = sjson.SetBytes(systemMessage, "content", instructions.String())
		appendMessage(systemMessage)
	}

	duplicateOutputIDs := make(map[string]struct{})

	// Convert input array to messages
	if input := root.Get("input"); input.Exists() && input.IsArray() {
		rawInputArray := input.Array()
		explicitOutputCounts := make(map[string]int)
		missingIDOutputsCount := 0
		for _, item := range rawInputArray {
			itemType := item.Get("type").String()
			if itemType == "function_call_output" || itemType == "custom_tool_call_output" {
				id := translatorcommon.ExtractResponsesCallID(item)
				if id != "" {
					explicitOutputCounts[id]++
				} else {
					missingIDOutputsCount++
				}
			}
		}

		unclaimedCalls := make(map[string]bool)
		for _, item := range rawInputArray {
			itemType := item.Get("type").String()
			if itemType == "function_call" || itemType == "custom_tool_call" {
				id := translatorcommon.ExtractResponsesCallID(item)
				if id != "" && explicitOutputCounts[id] == 0 {
					unclaimedCalls[id] = true
				}
			}
		}

		inputItems := translatorcommon.NormalizeResponsesToolCallOutputs(rawInputArray)
		if missingIDOutputsCount > 1 || (missingIDOutputsCount > 0 && len(unclaimedCalls) > 1) {
			for idx, item := range inputItems {
				itemType := item.Get("type").String()
				if itemType == "function_call_output" || itemType == "custom_tool_call_output" {
					if idx < len(rawInputArray) && translatorcommon.ExtractResponsesCallID(rawInputArray[idx]) == "" {
						raw := []byte(item.Raw)
						raw, _ = sjson.DeleteBytes(raw, "call_id")
						raw, _ = sjson.DeleteBytes(raw, "tool_call_id")
						raw, _ = sjson.DeleteBytes(raw, "callId")
						inputItems[idx] = gjson.ParseBytes(raw)
					}
				}
			}
		}

		hasReasoningInSession := false
		if effortRes := root.Get("reasoning.effort"); effortRes.Exists() {
			effort := strings.ToLower(strings.TrimSpace(effortRes.String()))
			if effort != "" && effort != "none" && effort != "0" && effort != "false" {
				hasReasoningInSession = true
			}
		} else if effortRes := root.Get("reasoning_effort"); effortRes.Exists() {
			effort := strings.ToLower(strings.TrimSpace(effortRes.String()))
			if effort != "" && effort != "none" && effort != "0" && effort != "false" {
				hasReasoningInSession = true
			}
		} else if reasoningObj := root.Get("reasoning"); reasoningObj.Exists() {
			reasoningRaw := strings.ToLower(strings.TrimSpace(reasoningObj.String()))
			if reasoningRaw != "" && reasoningRaw != "none" && reasoningRaw != "false" && reasoningRaw != "{}" {
				hasReasoningInSession = true
			}
		}
		if !hasReasoningInSession {
			for _, item := range rawInputArray {
				itemType := item.Get("type").String()
				if itemType == "reasoning" || item.Get("reasoning_content").Exists() {
					hasReasoningInSession = true
					break
				}
			}
		}

		pendingToolCalls := make([]interface{}, 0)
		pendingToolCallIDs := make([]string, 0)
		pendingReasoningContent := ""
		latestReasoningContent := ""
		awaitingToolOutputs := make(map[string]struct{})
		outputCounts := make(map[string]int)
		mergeableAssistantIndex := -1

		fallbackToolReasoning := func() string {
			if latestReasoningContent != "" {
				return latestReasoningContent
			}
			if hasReasoningInSession {
				return "[reasoning unavailable]"
			}
			return ""
		}

		takePendingReasoningContent := func() string {
			reasoningContent := pendingReasoningContent
			pendingReasoningContent = ""
			return reasoningContent
		}
		flushPendingToolCalls := func() {
			if len(pendingToolCalls) == 0 {
				return
			}

			reasoningContent := takePendingReasoningContent()
			mergedIntoAssistant := false
			if mergeableAssistantIndex >= 0 && mergeableAssistantIndex == len(messages)-1 {
				assistantMessage := gjson.ParseBytes(messages[mergeableAssistantIndex])
				if assistantMessage.Get("role").String() == "assistant" && !assistantMessage.Get("tool_calls").Exists() {
					updatedMessage, _ := sjson.SetBytes(messages[mergeableAssistantIndex], "tool_calls", pendingToolCalls)
					combinedReasoning := combineOpenAIResponsesReasoning(assistantMessage.Get("reasoning_content").String(), reasoningContent)
					if combinedReasoning != "" {
						updatedMessage, _ = sjson.SetBytes(updatedMessage, "reasoning_content", combinedReasoning)
						if isUsableResponsesReasoning(combinedReasoning) {
							latestReasoningContent = combinedReasoning
						}
					} else if fallback := fallbackToolReasoning(); fallback != "" {
						updatedMessage, _ = sjson.SetBytes(updatedMessage, "reasoning_content", fallback)
					}
					messages[mergeableAssistantIndex] = updatedMessage
					mergedIntoAssistant = true
				}
			}
			if !mergedIntoAssistant {
				assistantMessage := []byte(`{"role":"assistant","tool_calls":[]}`)
				assistantMessage, _ = sjson.SetBytes(assistantMessage, "tool_calls", pendingToolCalls)
				if reasoningContent != "" {
					assistantMessage, _ = sjson.SetBytes(assistantMessage, "reasoning_content", reasoningContent)
					if isUsableResponsesReasoning(reasoningContent) {
						latestReasoningContent = reasoningContent
					}
				} else if fallback := fallbackToolReasoning(); fallback != "" {
					assistantMessage, _ = sjson.SetBytes(assistantMessage, "reasoning_content", fallback)
				}
				appendMessage(assistantMessage)
			}
			for _, id := range pendingToolCallIDs {
				trimmed := strings.TrimSpace(id)
				if trimmed == "" {
					continue
				}
				awaitingToolOutputs[trimmed] = struct{}{}
			}
			pendingToolCalls = pendingToolCalls[:0]
			pendingToolCallIDs = pendingToolCallIDs[:0]
			mergeableAssistantIndex = -1
		}
		appendRegularMessage := func(message []byte) int {
			appendMessage(message)
			return len(messages) - 1
		}
		appendPendingReasoningMessage := func() {
			reasoningContent := takePendingReasoningContent()
			if reasoningContent == "" {
				return
			}
			if isUsableResponsesReasoning(reasoningContent) {
				latestReasoningContent = reasoningContent
			}
			message := []byte(`{"role":"assistant","content":"","reasoning_content":""}`)
			message, _ = sjson.SetBytes(message, "reasoning_content", reasoningContent)
			appendRegularMessage(message)
		}

		for _, item := range inputItems {
			itemType := item.Get("type").String()
			if itemType == "" && item.Get("role").String() != "" {
				itemType = "message"
			}
			if itemType != "function_call" && itemType != "custom_tool_call" {
				flushPendingToolCalls()
			}

			switch itemType {
			case "message", "":
				// Handle regular message conversion
				role := item.Get("role").String()
				if role == "developer" {
					role = "user"
				}
				mergeableAssistantIndex = -1
				if role != "assistant" {
					appendPendingReasoningMessage()
					latestReasoningContent = ""
				}
				message := []byte(`{"role":"","content":[]}`)
				message, _ = sjson.SetBytes(message, "role", role)

				if content := item.Get("content"); content.Exists() && content.IsArray() {
					var contentItems [][]byte
					content.ForEach(func(_, contentItem gjson.Result) bool {
						contentType := contentItem.Get("type").String()
						if contentType == "" {
							contentType = "input_text"
						}

						switch contentType {
						case "input_text", "output_text":
							text := contentItem.Get("text").String()
							contentPart := []byte(`{"type":"text","text":""}`)
							contentPart, _ = sjson.SetBytes(contentPart, "text", text)
							contentItems = append(contentItems, contentPart)
						case "input_video", "video_url":
							contentPart := []byte(`{"type":"video_url","video_url":{}}`)
							videoURL := contentItem.Get("video_url")
							if videoURL.IsObject() {
								contentPart, _ = sjson.SetRawBytes(contentPart, "video_url", []byte(videoURL.Raw))
							} else if videoURL.Exists() {
								contentPart, _ = sjson.SetRawBytes(contentPart, "video_url.url", []byte(videoURL.Raw))
							}
							if processing := contentItem.Get("processing"); processing.Exists() {
								contentPart, _ = sjson.SetRawBytes(contentPart, "video_url.processing", []byte(processing.Raw))
							}
							// Preserve malformed video parts for upstream validation instead of
							// silently turning a video request into a text-only request.
							contentItems = append(contentItems, contentPart)
						case "input_image":
							imageURL := contentItem.Get("image_url").String()
							contentPart := []byte(`{"type":"image_url","image_url":{"url":""}}`)
							contentPart, _ = sjson.SetBytes(contentPart, "image_url.url", imageURL)
							if detail, ok := normalizeChatImageDetail(contentItem.Get("detail")); ok && detail != "" {
								contentPart, _ = sjson.SetBytes(contentPart, "image_url.detail", detail)
							}
							contentItems = append(contentItems, contentPart)
						}
						return true
					})
					message = translatorcommon.SetRawArrayItems(message, "content", contentItems)
				} else if content.Type == gjson.String {
					message, _ = sjson.SetBytes(message, "content", content.String())
				}

				if role == "assistant" {
					reasoningContent := combineOpenAIResponsesReasoning(takePendingReasoningContent(), item.Get("reasoning_content").String())
					if reasoningContent != "" {
						message, _ = sjson.SetBytes(message, "reasoning_content", reasoningContent)
						if isUsableResponsesReasoning(reasoningContent) {
							latestReasoningContent = reasoningContent
						}
					}
				}

				messageIndex := appendRegularMessage(message)
				if role == "assistant" {
					mergeableAssistantIndex = messageIndex
				}

			case "reasoning":
				reasoningContent := collectOpenAIResponsesReasoningContent(item)
				pendingReasoningContent = combineOpenAIResponsesReasoning(pendingReasoningContent, reasoningContent)
				if isUsableResponsesReasoning(reasoningContent) {
					latestReasoningContent = reasoningContent
				}

			case "function_call":
				rc := item.Get("reasoning_content").String()
				pendingReasoningContent = combineOpenAIResponsesReasoning(pendingReasoningContent, rc)
				if isUsableResponsesReasoning(rc) {
					latestReasoningContent = rc
				}
				// Buffer consecutive function calls and emit them as one assistant message.
				toolCall := []byte(`{"id":"","type":"function","function":{"name":"","arguments":""}}`)

				if callId := translatorcommon.ExtractResponsesCallID(item); callId != "" {
					toolCall, _ = sjson.SetBytes(toolCall, "id", callId)
				}

				if name := item.Get("name"); name.Exists() {
					functionName := name.String()
					if namespace := strings.TrimSpace(item.Get("namespace").String()); namespace != "" {
						functionName = toolIndex.namespaceName(namespace, functionName)
					} else {
						functionName = toolIndex.canonicalName(functionName)
					}
					toolCall, _ = sjson.SetBytes(toolCall, "function.name", functionName)
				}

				if arguments := item.Get("arguments"); arguments.Exists() {
					toolCall, _ = sjson.SetBytes(toolCall, "function.arguments", arguments.String())
				}
				pendingToolCalls = append(pendingToolCalls, gjson.ParseBytes(toolCall).Value())
				if callID := translatorcommon.ExtractResponsesCallID(item); callID != "" {
					pendingToolCallIDs = append(pendingToolCallIDs, callID)
				}

			case "function_call_output":
				mergeableAssistantIndex = -1
				callID := translatorcommon.ExtractResponsesCallID(item)
				if callID != "" {
					outputCounts[callID]++
					if outputCounts[callID] > 1 {
						duplicateOutputIDs[callID] = struct{}{}
					}
				}
				if _, awaiting := awaitingToolOutputs[callID]; !awaiting {
					// Orphan outputs (empty call_id or no matching assistant
					// tool_calls, e.g. Codex send_message_to_thread cards) must
					// not become tool messages. Emit as user text instead.
					appendStandaloneResponsesToolOutputAsUser(item.Get("output"), setFunctionCallOutputContent, appendRegularMessage)
				} else {
					toolMessage := []byte(`{"role":"tool","tool_call_id":"","content":""}`)
					toolMessage, _ = sjson.SetBytes(toolMessage, "tool_call_id", callID)
					delete(awaitingToolOutputs, callID)
					if output := item.Get("output"); output.Exists() {
						var relayedImages [][]byte
						toolMessage, relayedImages = setFunctionCallOutputContent(toolMessage, output)
						relayedToolImages = append(relayedToolImages, relayedImages...)
					}
					appendMessage(toolMessage)
				}

			case "custom_tool_call":
				rc := item.Get("reasoning_content").String()
				pendingReasoningContent = combineOpenAIResponsesReasoning(pendingReasoningContent, rc)
				if isUsableResponsesReasoning(rc) {
					latestReasoningContent = rc
				}
				// Codex freeform tool call replay: wrap the raw input so it
				// matches the {"input": string} function shape used when
				// converting custom tool definitions.
				toolCall := []byte(`{"id":"","type":"function","function":{"name":"","arguments":""}}`)
				toolCall, _ = sjson.SetBytes(toolCall, "id", translatorcommon.ExtractResponsesCallID(item))
				functionName := item.Get("name").String()
				if namespace := item.Get("namespace").String(); namespace != "" {
					functionName = toolIndex.namespaceName(namespace, functionName)
				} else {
					functionName = toolIndex.canonicalName(functionName)
				}
				toolCall, _ = sjson.SetBytes(toolCall, "function.name", functionName)
				wrappedArgs, _ := sjson.SetBytes([]byte(`{"input":""}`), "input", item.Get("input").String())
				toolCall, _ = sjson.SetBytes(toolCall, "function.arguments", string(wrappedArgs))
				pendingToolCalls = append(pendingToolCalls, gjson.ParseBytes(toolCall).Value())
				if callID := translatorcommon.ExtractResponsesCallID(item); callID != "" {
					pendingToolCallIDs = append(pendingToolCallIDs, callID)
				}

			case "custom_tool_call_output":
				mergeableAssistantIndex = -1
				callID := translatorcommon.ExtractResponsesCallID(item)
				if callID != "" {
					outputCounts[callID]++
					if outputCounts[callID] > 1 {
						duplicateOutputIDs[callID] = struct{}{}
					}
				}
				if _, awaiting := awaitingToolOutputs[callID]; !awaiting {
					appendStandaloneResponsesToolOutputAsUser(item.Get("output"), setCustomToolCallOutputContent, appendRegularMessage)
				} else {
					toolMessage := []byte(`{"role":"tool","tool_call_id":"","content":""}`)
					toolMessage, _ = sjson.SetBytes(toolMessage, "tool_call_id", callID)
					delete(awaitingToolOutputs, callID)
					if output := item.Get("output"); output.Exists() {
						var relayedImages [][]byte
						toolMessage, relayedImages = setCustomToolCallOutputContent(toolMessage, output)
						relayedToolImages = append(relayedToolImages, relayedImages...)
					}
					appendMessage(toolMessage)
				}

			default:
				mergeableAssistantIndex = -1
			}

		}
		flushPendingToolCalls()
		appendPendingReasoningMessage()
		flushRelayImages()
	} else if input.Type == gjson.String {
		msg := []byte(`{}`)
		msg, _ = sjson.SetBytes(msg, "role", "user")
		msg, _ = sjson.SetBytes(msg, "content", input.String())
		appendMessage(msg)
	}

	if len(messages) > 0 {
		var extraAmbiguous []string
		for id := range duplicateOutputIDs {
			extraAmbiguous = append(extraAmbiguous, id)
		}
		messages = translatorcommon.AlignOpenAIToolCallMessages(messages, extraAmbiguous...)
		out, _ = sjson.SetRawBytes(out, "messages", translatorcommon.JoinRawArray(messages))
	}

	// Convert tools from responses format to chat completions format.
	// Codex Desktop (Responses Lite) delivers tool definitions through an
	// "additional_tools" input item instead of the top-level "tools" field,
	// so merge both sources.
	var chatCompletionsTools []interface{}
	for _, chatTool := range toolIndex.chatTools() {
		chatCompletionsTools = append(chatCompletionsTools, gjson.ParseBytes(chatTool).Value())
	}
	if len(chatCompletionsTools) > 0 {
		out, _ = sjson.SetBytes(out, "tools", chatCompletionsTools)
		if parallelToolCalls := root.Get("parallel_tool_calls"); parallelToolCalls.Exists() {
			out, _ = sjson.SetBytes(out, "parallel_tool_calls", parallelToolCalls.Bool())
		}
		if toolChoice := root.Get("tool_choice"); toolChoice.Exists() {
			out, _ = sjson.SetRawBytes(out, "tool_choice", convertResponsesToolChoiceWithIndex(toolChoice, toolIndex))
		}
	}

	if reasoningEffort := root.Get("reasoning.effort"); reasoningEffort.Exists() {
		effort := strings.ToLower(strings.TrimSpace(reasoningEffort.String()))
		if effort != "" {
			out, _ = sjson.SetBytes(out, "reasoning_effort", effort)
		}
	}

	return out
}

func convertResponsesToolChoiceToChatCompletions(toolChoice gjson.Result, inputRawJSON []byte) []byte {
	return convertResponsesToolChoiceWithIndex(toolChoice, newResponsesToolIndex(gjson.ParseBytes(inputRawJSON)))
}

func convertResponsesToolChoiceWithIndex(toolChoice gjson.Result, toolIndex *responsesToolIndex) []byte {
	if !toolChoice.IsObject() {
		return []byte(toolChoice.Raw)
	}

	choiceType := toolChoice.Get("type").String()
	if choiceType != "function" && choiceType != "custom" {
		return []byte(toolChoice.Raw)
	}

	name := toolChoice.Get("function.name").String()
	if name == "" {
		name = toolChoice.Get("custom.name").String()
	}
	if name == "" {
		name = toolChoice.Get("name").String()
	}
	if name == "" {
		return []byte(toolChoice.Raw)
	}

	namespace := strings.TrimSpace(toolChoice.Get("namespace").String())
	if namespace == "" {
		namespace = strings.TrimSpace(toolChoice.Get("function.namespace").String())
	}
	if namespace == "" {
		namespace = strings.TrimSpace(toolChoice.Get("custom.namespace").String())
	}
	if namespace != "" {
		name = toolIndex.namespaceName(namespace, name)
	} else {
		name = toolIndex.canonicalName(name)
	}

	converted := []byte(`{"type":"function","function":{"name":""}}`)
	converted, _ = sjson.SetBytes(converted, "function.name", name)
	return converted
}

func convertResponsesTextFormatToChatResponseFormat(textFormat gjson.Result) []byte {
	formatType := textFormat.Get("type").String()
	switch formatType {
	case "text", "json_object":
		responseFormat := []byte(`{"type":""}`)
		responseFormat, _ = sjson.SetBytes(responseFormat, "type", formatType)
		return responseFormat
	case "json_schema":
		responseFormat := []byte(`{"type":"json_schema","json_schema":{}}`)
		for _, field := range []string{"name", "description", "strict"} {
			if value := textFormat.Get(field); value.Exists() {
				responseFormat, _ = sjson.SetBytes(responseFormat, "json_schema."+field, value.Value())
			}
		}
		if schema := textFormat.Get("schema"); schema.Exists() {
			responseFormat, _ = sjson.SetRawBytes(responseFormat, "json_schema.schema", []byte(schema.Raw))
		}
		return responseFormat
	default:
		return nil
	}
}

// toolResultImagePlaceholder keeps the OpenAI tool message non-empty when a
// Responses tool output carried nothing but images.
const toolResultImagePlaceholder = "[Tool returned image content; the images follow in the next user message.]"

// toolResultImageRelayNotice labels the user message that carries relayed tool images.
const toolResultImageRelayNotice = "Images returned by the preceding tool call(s):"

func appendStandaloneResponsesToolOutputAsUser(output gjson.Result, setContent func([]byte, gjson.Result) ([]byte, [][]byte), appendMessage func([]byte) int) {
	userMessage := []byte(`{"role":"user","content":""}`)
	if output.Exists() {
		userMessage, _ = setContent(userMessage, output)
	}
	content := gjson.GetBytes(userMessage, "content")
	if !content.Exists() {
		return
	}
	if content.Type == gjson.String && strings.TrimSpace(content.String()) == "" {
		return
	}
	if content.IsArray() && !content.Get("0").Exists() {
		return
	}
	appendMessage(userMessage)
}

func setFunctionCallOutputContent(toolMessage []byte, output gjson.Result) ([]byte, [][]byte) {
	structuredContent := output
	if output.Type == gjson.String {
		if !gjson.Valid(output.String()) {
			toolMessage, _ = sjson.SetBytes(toolMessage, "content", output.String())
			return toolMessage, nil
		}
		structuredContent = gjson.Parse(output.String())
	}

	if hasChatToolOutputImagePart(structuredContent) {
		textParts := make([]string, 0, len(structuredContent.Array()))
		images := make([][]byte, 0, len(structuredContent.Array()))
		for _, item := range structuredContent.Array() {
			switch item.Get("type").String() {
			case "text", "input_text", "output_text":
				textParts = append(textParts, item.Get("text").String())
			case "image_url", "input_image":
				images = append(images, chatToolOutputContentPart(item))
			default:
				textParts = append(textParts, chatToolOutputFallbackText(item))
			}
		}
		content := strings.Join(textParts, "\n\n")
		if strings.TrimSpace(content) == "" {
			content = toolResultImagePlaceholder
		}
		toolMessage, _ = sjson.SetBytes(toolMessage, "content", content)
		return toolMessage, images
	}

	toolMessage, _ = sjson.SetBytes(toolMessage, "content", output.String())
	return toolMessage, nil
}

func setCustomToolCallOutputContent(toolMessage []byte, output gjson.Result) ([]byte, [][]byte) {
	structuredContent := output
	if output.Type == gjson.String && gjson.Valid(output.String()) {
		structuredContent = gjson.Parse(output.String())
	}
	if hasChatToolOutputImagePart(structuredContent) {
		return setFunctionCallOutputContent(toolMessage, output)
	}

	toolMessage, _ = sjson.SetBytes(toolMessage, "content", responsesToolOutputText(output))
	return toolMessage, nil
}

func chatToolOutputContentPart(item gjson.Result) []byte {
	itemType := item.Get("type").String()
	switch itemType {
	case "text", "input_text", "output_text":
		part := []byte(`{"type":"text","text":""}`)
		part, _ = sjson.SetBytes(part, "text", item.Get("text").String())
		return part
	case "image_url", "input_image":
		imageURL, detail, ok := chatToolOutputImageFields(item)
		if !ok {
			return chatToolOutputFallbackPart(item)
		}
		part := []byte(`{"type":"image_url","image_url":{"url":""}}`)
		part, _ = sjson.SetBytes(part, "image_url.url", imageURL)
		if detail != "" {
			part, _ = sjson.SetBytes(part, "image_url.detail", detail)
		}
		return part
	default:
		return chatToolOutputFallbackPart(item)
	}
}

func hasChatToolOutputImagePart(content gjson.Result) bool {
	if !content.IsArray() {
		return false
	}

	hasImage := false
	for _, item := range content.Array() {
		itemType := item.Get("type")
		if itemType.Type != gjson.String {
			continue
		}
		switch itemType.String() {
		case "text", "input_text", "output_text":
			if item.Get("text").Type != gjson.String {
				return false
			}
		case "image_url", "input_image":
			if _, _, ok := chatToolOutputImageFields(item); !ok {
				return false
			}
			hasImage = true
		}
	}
	return hasImage
}

func chatToolOutputImageFields(item gjson.Result) (imageURL, detail string, ok bool) {
	var imageURLValue gjson.Result
	var detailValue gjson.Result
	switch item.Get("type").String() {
	case "image_url":
		imageURLValue = item.Get("image_url.url")
		detailValue = item.Get("image_url.detail")
	case "input_image":
		imageURLValue = item.Get("image_url")
		detailValue = item.Get("detail")
	default:
		return "", "", false
	}

	if imageURLValue.Type != gjson.String {
		return "", "", false
	}
	imageURL = strings.TrimSpace(imageURLValue.String())
	if imageURL == "" {
		return "", "", false
	}

	detail, ok = normalizeChatImageDetail(detailValue)
	if !ok {
		return "", "", false
	}
	return imageURL, detail, true
}

func normalizeChatImageDetail(detailValue gjson.Result) (string, bool) {
	if !detailValue.Exists() {
		return "", true
	}
	if detailValue.Type != gjson.String {
		return "", false
	}

	normalizedDetail := strings.ToLower(strings.TrimSpace(detailValue.String()))
	switch normalizedDetail {
	case "auto", "low", "high":
		return normalizedDetail, true
	case "original":
		// Chat Completions does not support Codex's original detail value.
		return "high", true
	default:
		return "", true
	}
}

func chatToolOutputFallbackPart(item gjson.Result) []byte {
	part := []byte(`{"type":"text","text":""}`)
	part, _ = sjson.SetBytes(part, "text", chatToolOutputFallbackText(item))
	return part
}

func chatToolOutputFallbackText(item gjson.Result) string {
	text := item.Raw
	if item.Type == gjson.String || text == "" {
		text = item.String()
	}
	return text
}

// mergeRelayImagesIntoUserMessage prepends the relay notice and images to a user
// message so relayed tool images share the next user turn instead of creating a
// separate one.
func mergeRelayImagesIntoUserMessage(message []byte, images [][]byte) []byte {
	relayItems := make([][]byte, 0, len(images)+1)
	noticeJSON := []byte(`{"type":"text","text":""}`)
	noticeJSON, _ = sjson.SetBytes(noticeJSON, "text", toolResultImageRelayNotice)
	relayItems = append(relayItems, noticeJSON)
	relayItems = append(relayItems, images...)

	content := gjson.GetBytes(message, "content")
	if content.IsArray() {
		contentItems := make([][]byte, 0, len(relayItems)+int(content.Get("#").Int()))
		content.ForEach(func(_, item gjson.Result) bool {
			contentItems = append(contentItems, []byte(item.Raw))
			return true
		})
		return translatorcommon.SetRawArrayItems(message, "content", append(relayItems, contentItems...))
	}
	if content.Type == gjson.String && content.String() != "" {
		textPart := []byte(`{"type":"text","text":""}`)
		textPart, _ = sjson.SetBytes(textPart, "text", content.String())
		relayItems = append(relayItems, textPart)
	}
	return translatorcommon.SetRawArrayItems(message, "content", relayItems)
}

func collectOpenAIResponsesReasoningContent(item gjson.Result) string {
	var reasoningText strings.Builder
	if summary := item.Get("summary"); summary.Exists() && summary.IsArray() {
		summary.ForEach(func(_, summaryItem gjson.Result) bool {
			if summaryItem.Get("type").String() != "summary_text" {
				return true
			}
			reasoningText.WriteString(summaryItem.Get("text").String())
			return true
		})
	}
	if reasoningText.Len() == 0 {
		return "[reasoning unavailable]"
	}
	return reasoningText.String()
}

func combineOpenAIResponsesReasoning(existing, incoming string) string {
	existingTrimmed := strings.TrimSpace(existing)
	incomingTrimmed := strings.TrimSpace(incoming)

	switch {
	case existingTrimmed == "":
		return incoming
	case incomingTrimmed == "":
		return existing
	case existingTrimmed == "[reasoning unavailable]":
		return incoming
	case incomingTrimmed == "[reasoning unavailable]", existingTrimmed == incomingTrimmed:
		return existing
	default:
		return existing + "\n\n" + incoming
	}
}

func isUsableResponsesReasoning(reasoning string) bool {
	trimmed := strings.TrimSpace(reasoning)
	return trimmed != "" && trimmed != "[reasoning unavailable]"
}
