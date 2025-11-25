import Foundation

#if canImport(UIKit)
    import UIKit
    import CoreImage
#endif

#if canImport(AppKit)
    import AppKit
    import CoreImage
#endif

#if MLX
    import MLXLMCommon
    import MLX
    import MLXVLM
    import Tokenizers
    import Hub

    /// A language model that runs locally using MLX.
    ///
    /// Use this model to run language models on Apple silicon using the MLX framework.
    /// Models are automatically downloaded and cached when first used.
    ///
    /// ```swift
    /// let model = MLXLanguageModel(modelId: "mlx-community/Llama-3.2-3B-Instruct-4bit")
    /// ```
    public struct MLXLanguageModel: LanguageModel {
        /// The reason the model is unavailable.
        /// This model is always available.
        public typealias UnavailableReason = Never

        /// The model identifier from the MLX community on Hugging Face.
        public let modelId: String

        /// The optional Hub API instance for downloading models.
        public let hub: HubApi?

        /// Creates an MLX language model.
        ///
        /// - Parameters:
        ///   - modelId: The Hugging Face model identifier (for example, "mlx-community/Llama-3.2-3B-Instruct-4bit").
        ///   - hub: An optional Hub API instance for downloading models. If not provided, the default Hub API is used.
        public init(modelId: String, hub: HubApi? = nil) {
            self.modelId = modelId
            self.hub = hub
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

            let context: ModelContext
            if let hub = hub {
                context = try await loadModel(hub: hub, id: modelId)
            } else {
                context = try await loadModel(id: modelId)
            }

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
            let userSegments = extractPromptSegments(from: session, fallbackText: prompt.description)
            let userMessage = convertSegmentsToMLXMessage(userSegments)
            var chat: [MLXLMCommon.Chat.Message] = [userMessage]
            var allTextChunks: [String] = []
            var allEntries: [Transcript.Entry] = []

            // Loop until no more tool calls
            while true {
                // Build user input with current chat history and tools
                let userInput = MLXLMCommon.UserInput(
                    chat: chat,
                    processing: .init(resize: .init(width: 512, height: 512)),
                    tools: toolSpecs,
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

    // MARK: - Segment Extraction

    private func extractPromptSegments(from session: LanguageModelSession, fallbackText: String) -> [Transcript.Segment]
    {
        // Prefer the most recent Transcript.Prompt entry if present
        for entry in session.transcript.reversed() {
            if case .prompt(let p) = entry {
                return p.segments
            }
        }
        return [.text(.init(content: fallbackText))]
    }

    private func convertSegmentsToMLXMessage(_ segments: [Transcript.Segment]) -> MLXLMCommon.Chat.Message {
        var textParts: [String] = []
        var images: [MLXLMCommon.UserInput.Image] = []

        for segment in segments {
            switch segment {
            case .text(let text):
                textParts.append(text.content)
            case .structure(let structured):
                textParts.append(structured.content.jsonString)
            case .image(let imageSegment):
                switch imageSegment.source {
                case .url(let url):
                    images.append(.url(url))
                case .data(let data, _):
                    #if canImport(UIKit)
                        if let uiImage = UIKit.UIImage(data: data),
                            let ciImage = CIImage(image: uiImage)
                        {
                            images.append(.ciImage(ciImage))
                        }
                    #elseif canImport(AppKit)
                        if let nsImage = AppKit.NSImage(data: data),
                            let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil)
                        {
                            let ciImage = CIImage(cgImage: cgImage)
                            images.append(.ciImage(ciImage))
                        }
                    #endif
                }
            }
        }

        let content = textParts.joined(separator: "\n")
        return MLXLMCommon.Chat.Message(role: .user, content: content, images: images)
    }

    // MARK: - Tool Conversion

    private func convertToolToMLXSpec(_ tool: any Tool) -> ToolSpec {
        // Convert AnyLanguageModel's GenerationSchema to JSON-compatible dictionary
        let parametersDict: [String: Any]
        do {
            let resolvedSchema = tool.parameters.withResolvedRoot() ?? tool.parameters
            let encoder = JSONEncoder()
            let data = try encoder.encode(resolvedSchema)
            if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
                parametersDict = json
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
            case .image:
                // Image segments are not supported in MLX tool output
                break
            }
        }
        return textParts.joined(separator: "\n")
    }
#endif  // MLX
