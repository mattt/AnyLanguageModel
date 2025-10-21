import Foundation
import MLXLMCommon
import Tokenizers

/// A language model that runs locally using MLX.
///
/// Use this model to run language models on Apple silicon using the MLX framework.
/// Models are automatically downloaded and cached when first used.
///
/// ```swift
/// let model = MLXLanguageModel(modelId: "mlx-community/Llama-3.2-3B-Instruct-4bit")
/// ```
public struct MLXLanguageModel: LanguageModel {
    /// The model identifier from the MLX community on Hugging Face.
    public let modelId: String

    /// Creates an MLX language model.
    ///
    /// - Parameter modelId: The Hugging Face model identifier (for example, "mlx-community/Llama-3.2-3B-Instruct-4bit").
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

        // Map AnyLanguageModel GenerationOptions to MLX GenerateParameters
        let generateParameters = toGenerateParameters(options)

        // Start with user prompt
        var chat: [MLXLMCommon.Chat.Message] = [.user(prompt.description)]
        var allTextChunks: [String] = []
        var allEntries: [Transcript.Entry] = []

        // Loop until no more tool calls
        while true {
            // Build user input with current chat history and tools
            let userInput = MLXLMCommon.UserInput(
                chat: chat,
                tools: toolSpecs
            )
            let lmInput = try await context.processor.prepare(input: userInput)

            // Generate
            let stream = try MLXLMCommon.generate(
                input: lmInput,
                parameters: generateParameters,
                context: context
            )

            var chunks: [String] = []
            var collectedToolCalls: [MLXLMCommon.ToolCall] = []

            for await item in stream {
                switch item {
                case .chunk(let text):
                    chunks.append(text)
                case .info:
                    break
                case .toolCall(let call):
                    collectedToolCalls.append(call)
                }
            }

            let assistantText = chunks.joined()
            allTextChunks.append(assistantText)

            // Add assistant response to chat history
            if !assistantText.isEmpty {
                chat.append(.assistant(assistantText))
            }

            // If there are tool calls, execute them and continue
            if !collectedToolCalls.isEmpty {
                let invocations = try await resolveToolCalls(collectedToolCalls, session: session)
                if !invocations.isEmpty {
                    allEntries.append(.toolCalls(Transcript.ToolCalls(invocations.map(\.call))))

                    // Execute each tool and add results to chat
                    for invocation in invocations {
                        allEntries.append(.toolOutput(invocation.output))

                        // Convert tool output to JSON string for MLX
                        let toolResultJSON = toolOutputToJSON(invocation.output)
                        chat.append(.tool(toolResultJSON))
                    }

                    // Continue loop to generate with tool results
                    continue
                }
            }

            // No more tool calls, exit loop
            break
        }

        let text = allTextChunks.joined()
        return LanguageModelSession.Response(
            content: text as! Content,
            rawContent: GeneratedContent(text),
            transcriptEntries: ArraySlice(allEntries)
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

// TODO: Improve JSON handling by using JSONValue from JSONSchema package
// instead of [String: Any] for better type safety
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

private func toolOutputToJSON(_ output: Transcript.ToolOutput) -> String {
    // Extract text content from segments
    var textParts: [String] = []
    for segment in output.segments {
        switch segment {
        case .text(let textSegment):
            textParts.append(textSegment.content)
        case .structure(let structuredSegment):
            // structured content already has jsonString property
            textParts.append(structuredSegment.content.jsonString)
        }
    }
    return textParts.joined(separator: "\n")
}
