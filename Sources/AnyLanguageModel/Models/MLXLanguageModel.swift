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

        /// The model identifier.
        public let modelId: String

        /// The Hub API instance for downloading models.
        public let hub: HubApi?

        /// The local directory containing the model files.
        public let directory: URL?

        /// Creates an MLX language model.
        ///
        /// - Parameters:
        ///   - modelId: The model identifier (for example, "mlx-community/Llama-3.2-3B-Instruct-4bit").
        ///   - hub: An optional Hub API instance for downloading models. If not provided, the default Hub API is used.
        ///   - directory: An optional local directory URL containing the model files. If provided, the model is loaded from this directory instead of downloading.
        public init(modelId: String, hub: HubApi? = nil, directory: URL? = nil) {
            self.modelId = modelId
            self.hub = hub
            self.directory = directory
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
            if let directory {
                context = try await loadModel(directory: directory)
            } else if let hub {
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

            // Build chat history from full transcript
            var chat = convertTranscriptToMLXChat(session: session, fallbackPrompt: prompt.description)

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
            guard type == String.self else {
                fatalError("MLXLanguageModel only supports generating String content")
            }

            let modelId = self.modelId
            let hub = self.hub
            let directory = self.directory

            let stream: AsyncThrowingStream<LanguageModelSession.ResponseStream<Content>.Snapshot, any Error> = .init {
                continuation in
                let task = Task { @Sendable in
                    do {
                        let context: ModelContext
                        if let directory {
                            context = try await loadModel(directory: directory)
                        } else if let hub {
                            context = try await loadModel(hub: hub, id: modelId)
                        } else {
                            context = try await loadModel(id: modelId)
                        }

                        let generateParameters = toGenerateParameters(options)

                        // Build chat history from full transcript
                        let chat = convertTranscriptToMLXChat(session: session, fallbackPrompt: prompt.description)

                        let userInput = MLXLMCommon.UserInput(
                            chat: chat,
                            processing: .init(resize: .init(width: 512, height: 512)),
                            tools: nil
                        )
                        let lmInput = try await context.processor.prepare(input: userInput)

                        let mlxStream = try MLXLMCommon.generate(
                            input: lmInput,
                            parameters: generateParameters,
                            context: context
                        )

                        var accumulatedText = ""
                        for await item in mlxStream {
                            if Task.isCancelled { break }

                            switch item {
                            case .chunk(let text):
                                accumulatedText += text
                                let raw = GeneratedContent(accumulatedText)
                                let content: Content.PartiallyGenerated = (accumulatedText as! Content)
                                    .asPartiallyGenerated()
                                continuation.yield(.init(content: content, rawContent: raw))
                            case .info, .toolCall:
                                break
                            }
                        }

                        continuation.finish()
                    } catch {
                        continuation.finish(throwing: error)
                    }
                }
                continuation.onTermination = { _ in task.cancel() }
            }

            return LanguageModelSession.ResponseStream(stream: stream)
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

    // MARK: - Transcript Conversion

    private func convertTranscriptToMLXChat(
        session: LanguageModelSession,
        fallbackPrompt: String
    ) -> [MLXLMCommon.Chat.Message] {
        var chat: [MLXLMCommon.Chat.Message] = []

        // Check if instructions are already in transcript
        let hasInstructionsInTranscript = session.transcript.contains {
            if case .instructions = $0 { return true }
            return false
        }

        // Add instructions from session if present and not in transcript
        if !hasInstructionsInTranscript,
            let instructions = session.instructions?.description,
            !instructions.isEmpty
        {
            chat.append(.init(role: .system, content: instructions))
        }

        // Convert each transcript entry
        for entry in session.transcript {
            switch entry {
            case .instructions(let instr):
                chat.append(makeMLXChatMessage(from: instr.segments, role: .system))

            case .prompt(let prompt):
                chat.append(makeMLXChatMessage(from: prompt.segments, role: .user))

            case .response(let response):
                let content = response.segments.map { extractText(from: $0) }.joined(separator: "\n")
                chat.append(.assistant(content))

            case .toolCalls:
                // Tool calls are handled inline during generation loop
                break

            case .toolOutput(let toolOutput):
                let content = toolOutput.segments.map { extractText(from: $0) }.joined(separator: "\n")
                chat.append(.tool(content))
            }
        }

        // If no user message in transcript, add fallback prompt
        let hasUserMessage = chat.contains { $0.role == .user }
        if !hasUserMessage {
            chat.append(.init(role: .user, content: fallbackPrompt))
        }

        return chat
    }

    private func extractText(from segment: Transcript.Segment) -> String {
        switch segment {
        case .text(let text):
            return text.content
        case .structure(let structured):
            return structured.content.jsonString
        case .image:
            return ""
        }
    }

    private func makeMLXChatMessage(
        from segments: [Transcript.Segment],
        role: MLXLMCommon.Chat.Message.Role
    ) -> MLXLMCommon.Chat.Message {
        var textParts: [String] = []
        var images: [MLXLMCommon.UserInput.Image] = []

        for segment in segments {
            switch segment {
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
            default:
                let text = extractText(from: segment)
                if !text.isEmpty {
                    textParts.append(text)
                }
            }
        }

        let content = textParts.joined(separator: "\n")
        return MLXLMCommon.Chat.Message(role: role, content: content, images: images)
    }

    // MARK: - Tool Conversion

    private func convertToolToMLXSpec(_ tool: any Tool) -> ToolSpec {
        // Convert AnyLanguageModel's GenerationSchema to Sendable dictionary
        // using MLXLMCommon.JSONValue which is already Sendable
        let parametersValue: JSONValue
        do {
            let resolvedSchema = tool.parameters.withResolvedRoot() ?? tool.parameters
            let data = try JSONEncoder().encode(resolvedSchema)
            parametersValue = try JSONDecoder().decode(JSONValue.self, from: data)
        } catch {
            parametersValue = .object(["type": .string("object"), "properties": .object([:]), "required": .array([])])
        }

        return [
            "type": "function",
            "function": [
                "name": tool.name,
                "description": tool.description,
                "parameters": parametersValue,
            ] as [String: any Sendable],
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
