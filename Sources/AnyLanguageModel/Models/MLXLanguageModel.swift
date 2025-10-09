import Foundation
import MLXLMCommon

public struct MLXLanguageModel: LanguageModel {
    public let modelId: String

    public init(modelId: String) {
        self.modelId = modelId
    }

    public func respond<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
        // For now, only String is supported
        guard type == String.self else {
            fatalError("MLXLanguageModel only supports generating String content")
        }

        // Load model context via MLXLMCommon
        let context = try await MLXLMCommon.loadModel(id: modelId)

        // Build user input from prompt only
        let userInput = MLXLMCommon.UserInput(chat: [.user(prompt.description)])
        let lmInput = try await context.processor.prepare(input: userInput)

        // Map AnyLanguageModel GenerationOptions to MLX GenerateParameters
        let generateParameters = toGenerateParameters(options)

        // Use streaming generation to capture tool calls
        let stream = try MLXLMCommon.generate(
            input: lmInput,
            parameters: generateParameters,
            context: context
        )

        var chunks: [String] = []
        var collectedToolCalls: [MLXLMCommon.ToolCall] = []

        for await item in stream {
            if let chunk = item.chunk {
                chunks.append(chunk)
            }
            if let toolCall = item.toolCall {
                collectedToolCalls.append(toolCall)
            }
        }

        var entries: [Transcript.Entry] = []
        if !collectedToolCalls.isEmpty {
            let invocations = try await resolveToolCalls(collectedToolCalls, session: session)
            if !invocations.isEmpty {
                entries.append(.toolCalls(Transcript.ToolCalls(invocations.map(\.call))))
                for invocation in invocations {
                    entries.append(.toolOutput(invocation.output))
                }
            }
        }

        let text = chunks.joined()
        return LanguageModelSession.Response(
            content: text as! Content,
            rawContent: GeneratedContent(text),
            transcriptEntries: ArraySlice(entries)
        )
    }

    public func streamResponse<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) -> sending LanguageModelSession.ResponseStream<Content> where Content: Generable {
        // For now, only String is supported
        guard type == String.self else {
            fatalError("MLXLanguageModel only supports generating String content")
        }

        // Streaming API in AnyLanguageModel currently yields once; return an empty snapshot
        let empty = ""
        return LanguageModelSession.ResponseStream(
            content: empty as! Content,
            rawContent: GeneratedContent(empty)
        )
    }
}

// MARK: - Options Mapping

private func toGenerateParameters(_ options: GenerationOptions) -> MLXLMCommon.GenerateParameters {
    MLXLMCommon.GenerateParameters(
        maxTokens: options.maximumResponseTokens,
        maxKVSize: nil,
        kvBits: nil,
        kvGroupSize: 64,
        quantizedKVStart: 0,
        temperature: Float(options.temperature ?? 0.6),
        topP: 1.0,
        repetitionPenalty: nil,
        repetitionContextSize: 20
    )
}

// MARK: - Tool Invocation Handling

private struct ToolInvocationResult {
    let call: Transcript.ToolCall
    let output: Transcript.ToolOutput
}

private func resolveToolCalls(
    _ toolCalls: [MLXLMCommon.ToolCall],
    session: LanguageModelSession
) async throws -> [ToolInvocationResult] {
    if toolCalls.isEmpty { return [] }

    var toolsByName: [String: any Tool] = [:]
    for tool in session.tools {
        if toolsByName[tool.name] == nil {
            toolsByName[tool.name] = tool
        }
    }

    var results: [ToolInvocationResult] = []
    results.reserveCapacity(toolCalls.count)

    for call in toolCalls {
        let args = try toGeneratedContent(call.function.arguments)
        let callID = UUID().uuidString
        let transcriptCall = Transcript.ToolCall(
            id: callID,
            toolName: call.function.name,
            arguments: args
        )

        guard let tool = toolsByName[call.function.name] else {
            let message = Transcript.Segment.text(.init(content: "Tool not found: \(call.function.name)"))
            let output = Transcript.ToolOutput(
                id: callID,
                toolName: call.function.name,
                segments: [message]
            )
            results.append(ToolInvocationResult(call: transcriptCall, output: output))
            continue
        }

        do {
            let segments = try await tool.makeOutputSegments(from: args)
            let output = Transcript.ToolOutput(
                id: tool.name,
                toolName: tool.name,
                segments: segments
            )
            results.append(ToolInvocationResult(call: transcriptCall, output: output))
        } catch {
            throw LanguageModelSession.ToolCallError(tool: tool, underlyingError: error)
        }
    }

    return results
}

private func toGeneratedContent(_ args: [String: MLXLMCommon.JSONValue]) throws -> GeneratedContent {
    let data = try JSONEncoder().encode(args)
    let json = String(data: data, encoding: .utf8) ?? "{}"
    return try GeneratedContent(json: json)
}
