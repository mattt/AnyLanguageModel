#if canImport(FoundationModels) && canImport(CoreAILanguageModels) && canImport(CoreAI) && compiler(>=6.4)
    import CoreAILanguageModels
    import Foundation
    import FoundationModels

    /// A language model that runs Core AI model bundles on Apple's production inference layer.
    ///
    /// Use this model to run open-weights language models exported to the Core AI
    /// `.aimodel` format entirely on-device, through the same session machinery as
    /// Apple's Foundation Models.
    ///
    /// ```swift
    /// let model = CoreAILanguageModel(resourcesAt: modelURL)
    /// ```
    @available(macOS 27.0, iOS 27.0, *)
    public actor CoreAILanguageModel: LanguageModel {
        /// The reason the model is unavailable.
        public enum UnavailableReason: Sendable {
            case resourcesNotFound
        }

        /// The location of the Core AI resources folder for this model.
        nonisolated public let url: URL

        private let variant: String?
        private var underlying: CoreAILanguageModels.CoreAILanguageModel?

        /// Creates a Core AI language model from a resources folder.
        ///
        /// - Parameters:
        ///   - url: The folder containing the `.aimodel` bundle, tokenizer, and metadata.
        ///   - variant: An optional bundle variant to load.
        public init(resourcesAt url: URL, variant: String? = nil) {
            self.url = url
            self.variant = variant
        }

        /// The availability status for this model.
        nonisolated public var availability: Availability<UnavailableReason> {
            if FileManager.default.fileExists(atPath: url.path) {
                return .available
            }
            return .unavailable(.resourcesNotFound)
        }

        /// Loads the model resources eagerly.
        public func load() async throws {
            try await underlyingModel().load()
        }

        /// Unloads the model resources.
        public func unload() async {
            underlying?.unload()
            underlying = nil
        }

        private func underlyingModel() async throws -> CoreAILanguageModels.CoreAILanguageModel {
            if let underlying {
                return underlying
            }
            let model = try await CoreAILanguageModels.CoreAILanguageModel(
                resourcesAt: url,
                variant: variant
            )
            underlying = model
            return model
        }

        nonisolated public func respond<Content>(
            within session: LanguageModelSession,
            to prompt: Prompt,
            generating type: Content.Type,
            includeSchemaInPrompt: Bool,
            options: GenerationOptions
        ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
            let fmTools = session.tools.toFoundationModels()
            let fmTranscript = fmTranscriptDroppingDuplicatePrompt(session.transcript, prompt: prompt).toFoundationModels(
                instructions: session.instructions,
                toolDefinitions: session.tools
                    .filter(\.includesSchemaInInstructions)
                    .map { Transcript.ToolDefinition(tool: $0) }
            )
            return try await fmRespond(
                makeSession: {
                    FoundationModels.LanguageModelSession(
                        model: try await self.underlyingModel(),
                        tools: fmTools,
                        transcript: fmTranscript
                    )
                },
                fmPrompt: prompt.toFoundationModels(),
                fmOptions: options.toFoundationModels(),
                type: type,
                includeSchemaInPrompt: includeSchemaInPrompt
            )
        }

        nonisolated public func streamResponse<Content>(
            within session: LanguageModelSession,
            to prompt: Prompt,
            generating type: Content.Type,
            includeSchemaInPrompt: Bool,
            options: GenerationOptions
        ) -> sending LanguageModelSession.ResponseStream<Content> where Content: Generable {
            let fmTools = session.tools.toFoundationModels()
            let fmTranscript = fmTranscriptDroppingDuplicatePrompt(session.transcript, prompt: prompt).toFoundationModels(
                instructions: session.instructions,
                toolDefinitions: session.tools
                    .filter(\.includesSchemaInInstructions)
                    .map { Transcript.ToolDefinition(tool: $0) }
            )
            return fmStreamResponse(
                makeSession: {
                    FoundationModels.LanguageModelSession(
                        model: try await self.underlyingModel(),
                        tools: fmTools,
                        transcript: fmTranscript
                    )
                },
                fmPrompt: prompt.toFoundationModels(),
                fmOptions: options.toFoundationModels(),
                type: type,
                includeSchemaInPrompt: includeSchemaInPrompt
            )
        }

        nonisolated public func logFeedbackAttachment(
            within session: LanguageModelSession,
            sentiment: LanguageModelFeedback.Sentiment?,
            issues: [LanguageModelFeedback.Issue],
            desiredOutput: Transcript.Entry?
        ) -> Data {
            Data()
        }
    }
#endif
