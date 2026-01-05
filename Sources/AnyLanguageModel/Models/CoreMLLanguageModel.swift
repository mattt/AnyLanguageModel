#if CoreML
    import Foundation
    import CoreML
    import Tokenizers
    @preconcurrency import Generation
    @preconcurrency import Models

    /// A language model that runs locally using Core ML.
    ///
    /// Use this model to run language models on-device with Core ML.
    /// The model must be compiled to `.mlmodelc` format before use.
    ///
    /// ```swift
    /// let modelURL = Bundle.main.url(
    ///     forResource: "MyModel",
    ///     withExtension: "mlmodelc"
    /// )!
    /// let model = try await CoreMLLanguageModel(url: modelURL)
    /// ```
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    public struct CoreMLLanguageModel: AnyLanguageModel.LanguageModel {
        /// The reason the model is unavailable.
        /// This model is always available.
        public typealias UnavailableReason = Never

        private let model: Models.LanguageModel
        private let tokenizer: any Tokenizer
        private let chatTemplateHandler: (@Sendable (Instructions?, Prompt) -> [Message])?
        private let toolsHandler: (@Sendable ([any Tool]) -> [ToolSpec])?

        /// Creates a Core ML language model.
        ///
        /// - Parameters:
        ///   - url: The URL to a compiled Core ML model (`.mlmodelc`).
        ///   - computeUnits: The compute units to use for inference.
        ///   - chatTemplateHandler: An optional handler to format chat messages.
        ///   - toolsHandler: An optional handler to convert tools to the model's expected format.
        ///
        /// - Throws: A `CoreMLLanguageModelError` if the model can't be loaded, the file doesn't exist, or the model is invalid.
        public init(
            url: URL,
            computeUnits: MLComputeUnits = .all,
            chatTemplateHandler: (@Sendable (Instructions?, Prompt) -> [Message])? = nil,
            toolsHandler: (@Sendable ([any Tool]) -> [ToolSpec])? = nil
        ) async throws {
            // Ensure the model is already compiled
            guard url.pathExtension == "mlmodelc" else {
                throw CoreMLLanguageModelError.compiledModelRequired
            }

            // Check if the file exists first
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw CoreMLLanguageModelError.modelNotFound(url)
            }

            do {
                // Load the model with the specified compute units
                self.model = try Models.LanguageModel.loadCompiled(url: url, computeUnits: computeUnits)
            } catch {
                // Map CoreML errors to our specific error cases
                throw CoreMLLanguageModelError.modelInvalid(url, underlyingError: error)
            }

            // Load the tokenizer
            self.tokenizer = try await model.tokenizer

            self.chatTemplateHandler = chatTemplateHandler
            self.toolsHandler = toolsHandler
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
                fatalError("CoreMLLanguageModel only supports generating String content")
            }

            try validateNoImageSegments(in: session)

            // Convert AnyLanguageModel GenerationOptions to swift-transformers GenerationConfig
            let generationConfig = toGenerationConfig(options)

            let tokens: [Int]
            if let chatTemplateHandler = chatTemplateHandler {
                // Use chat template handler with optional tools
                let messages = chatTemplateHandler(session.instructions, prompt)
                let toolSpecs: [ToolSpec]? = toolsHandler?(session.tools)
                tokens = try tokenizer.applyChatTemplate(messages: messages, tools: toolSpecs)
            } else {
                // Fall back to direct tokenizer encoding
                tokens = tokenizer.encode(text: prompt.description)
            }

            // Reset model state for new generation
            await model.resetState()

            let response = await model.generate(
                config: generationConfig,
                tokens: tokens,
                model: model.callAsFunction
            )

            return LanguageModelSession.Response(
                content: response as! Content,
                rawContent: GeneratedContent(response),
                transcriptEntries: ArraySlice([])
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
                fatalError("CoreMLLanguageModel only supports generating String content")
            }

            // Validate that no image segments are present
            do {
                try validateNoImageSegments(in: session)
            } catch {
                return LanguageModelSession.ResponseStream(
                    stream: AsyncThrowingStream { continuation in
                        continuation.finish(throwing: error)
                    }
                )
            }

            // Convert AnyLanguageModel GenerationOptions to swift-transformers GenerationConfig
            let generationConfig = toGenerationConfig(options)

            // Transform the generation into ResponseStream snapshots
            let stream: AsyncThrowingStream<LanguageModelSession.ResponseStream<Content>.Snapshot, any Error> = .init {
                @Sendable continuation in
                let task = Task {
                    do {
                        let tokens: [Int]
                        if let chatTemplateHandler = chatTemplateHandler {
                            // Use chat template handler with optional tools
                            let messages = chatTemplateHandler(session.instructions, prompt)
                            let toolSpecs: [ToolSpec]? = toolsHandler?(session.tools)
                            tokens = try tokenizer.applyChatTemplate(messages: messages, tools: toolSpecs)
                        } else {
                            // Fall back to direct tokenizer encoding
                            tokens = tokenizer.encode(text: prompt.description)
                        }

                        await model.resetState()

                        // Decode the rendered prompt once to strip it from streamed output
                        let promptTextPrefix = tokenizer.decode(tokens: tokens)
                        var accumulatedText = ""

                        _ = await model.generate(
                            config: generationConfig,
                            tokens: tokens,
                            model: model.callAsFunction
                        ) { tokenIds in
                            // Decode full text and strip the prompt prefix
                            let fullText = tokenizer.decode(tokens: tokenIds)
                            let assistantText: String
                            if fullText.hasPrefix(promptTextPrefix) {
                                let startIdx = fullText.index(fullText.startIndex, offsetBy: promptTextPrefix.count)
                                assistantText = String(fullText[startIdx...])
                            } else {
                                assistantText = fullText
                            }

                            // Compute delta vs accumulated text and yield
                            if assistantText.count >= accumulatedText.count,
                                assistantText.hasPrefix(accumulatedText)
                            {
                                let startIdx = assistantText.index(
                                    assistantText.startIndex,
                                    offsetBy: accumulatedText.count
                                )
                                let delta = String(assistantText[startIdx...])
                                accumulatedText += delta
                            } else {
                                accumulatedText = assistantText
                            }

                            continuation.yield(
                                .init(
                                    content: (accumulatedText as! Content).asPartiallyGenerated(),
                                    rawContent: GeneratedContent(accumulatedText)
                                )
                            )
                        }

                        continuation.finish()
                    } catch {
                        continuation.finish(throwing: error)
                    }
                }

                continuation.onTermination = { _ in
                    task.cancel()
                }
            }

            return LanguageModelSession.ResponseStream(stream: stream)
        }

        // MARK: - Image Validation

        private func validateNoImageSegments(in session: LanguageModelSession) throws {
            // Note: Instructions is a plain text type without segments, so no image check needed there.
            // Check for image segments in the most recent prompt
            for entry in session.transcript.reversed() {
                if case .prompt(let p) = entry {
                    for segment in p.segments {
                        if case .image = segment {
                            throw CoreMLLanguageModelError.unsupportedFeature
                        }
                    }
                    break
                }
            }
        }
    }

    /// Errors that can occur when working with Core ML language models.
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    public enum CoreMLLanguageModelError: LocalizedError {
        /// The provided model isn't a compiled Core ML model.
        case compiledModelRequired

        /// The model file was not found at the specified URL.
        case modelNotFound(URL)

        /// The model file was found but is corrupted, incompatible, or otherwise invalid.
        case modelInvalid(URL, underlyingError: Error)
        /// Image segments are not supported in CoreMLLanguageModel
        case unsupportedFeature

        public var errorDescription: String? {
            switch self {
            case .compiledModelRequired:
                return
                    "A compiled Core ML model (.mlmodelc) is required. Please compile your model first using MLModel.compileModel(at:)."
            case .modelNotFound(let url):
                return "Core ML model not found at: \(url.path). Please verify the file exists and the path is correct."
            case .modelInvalid(let url, let underlyingError):
                return
                    "Core ML model at \(url.path) is invalid or corrupted: \(underlyingError.localizedDescription). Please verify the model file is valid and compatible with the current Core ML version."
            case .unsupportedFeature:
                return "This CoreMLLanguageModel does not support image segments"
            }
        }
    }

    // MARK: -

    private func toGenerationConfig(_ options: GenerationOptions) -> GenerationConfig {
        var config = GenerationConfig(maxNewTokens: options.maximumResponseTokens ?? 2048)

        // Map temperature
        if let temperature = options.temperature {
            config.temperature = Float(temperature)
        }

        // Map sampling mode
        if let sampling = options.sampling {
            switch sampling.mode {
            case .greedy:
                config.doSample = false
            case .topK(let k, _):
                config.doSample = true
                config.topK = k
            case .nucleus(let p, _):
                config.doSample = true
                config.topP = Float(p)
            }
        }

        return config
    }
#endif  // CoreML
