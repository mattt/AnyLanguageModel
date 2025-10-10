import Foundation
import MLXLMCommon
import Tokenizers

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

        // Convert session tools to MLX ToolSpec format
        let toolSpecs: [ToolSpec]? =
            session.tools.isEmpty
            ? nil
            : session.tools.map { tool in
                convertToolToMLXSpec(tool)
            }

        // Build user input from prompt and tools
        let userInput = MLXLMCommon.UserInput(
            chat: [.user(prompt.description)],
            tools: toolSpecs
        )
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

// MARK: - Tool Conversion

private func convertToolToMLXSpec(_ tool: any Tool) -> ToolSpec {
    // Convert AnyLanguageModel's GenerationSchema to JSON-compatible dictionary
    let parametersDict: [String: Any]
    do {
        let encoder = JSONEncoder()
        let data = try encoder.encode(tool.parameters)
        if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
            // Resolve $ref if present
            if let ref = json["$ref"] as? String,
                let defs = json["$defs"] as? [String: Any]
            {
                // Extract the definition name from the $ref (e.g., "#/$defs/TypeName" -> "TypeName")
                let defName = ref.replacingOccurrences(of: "#/$defs/", with: "")
                if let resolvedDef = defs[defName] as? [String: Any] {
                    parametersDict = resolvedDef
                } else {
                    parametersDict = ["type": "object", "properties": [:], "required": []]
                }
            } else {
                parametersDict = json
            }
        } else {
            parametersDict = ["type": "object", "properties": [:], "required": []]
        }
    } catch {
        parametersDict = ["type": "object", "properties": [:], "required": []]
    }

    return [
        "type": "function",
        "function": [
            "name": tool.name,
            "description": tool.description,
            "parameters": parametersDict,
        ],
    ]
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
