#if canImport(FoundationModels) && compiler(>=6.4)
    import Foundation
    import FoundationModels

    /// A language model that uses Apple's Private Cloud Compute.
    ///
    /// Use this model to generate text using Apple's larger server-hosted models,
    /// running on Apple silicon servers under the Private Cloud Compute privacy
    /// architecture. Requests are stateless and cryptographically attested, and
    /// no data is retained.
    ///
    /// ```swift
    /// let model = PrivateCloudComputeLanguageModel()
    /// ```
    @available(macOS 27.0, iOS 27.0, *)
    public struct PrivateCloudComputeLanguageModel: LanguageModel {
        /// The reason the model is unavailable.
        public typealias UnavailableReason = FoundationModels.PrivateCloudComputeLanguageModel.Availability
            .UnavailableReason

        let pccModel: FoundationModels.PrivateCloudComputeLanguageModel

        /// The default Private Cloud Compute language model.
        public static var `default`: PrivateCloudComputeLanguageModel {
            PrivateCloudComputeLanguageModel()
        }

        /// Creates the default Private Cloud Compute language model.
        public init() {
            self.pccModel = FoundationModels.PrivateCloudComputeLanguageModel()
        }

        /// The current quota usage for Private Cloud Compute requests.
        public var quotaUsage: FoundationModels.PrivateCloudComputeLanguageModel.QuotaUsage {
            pccModel.quotaUsage
        }

        /// Whether the model accepts image input.
        public var supportsImageInput: Bool {
            pccModel.capabilities.contains(.vision)
        }

        /// The availability status for the Private Cloud Compute language model.
        public var availability: Availability<UnavailableReason> {
            switch pccModel.availability {
            case .available:
                .available
            case .unavailable(let reason):
                .unavailable(reason)
            }
        }

        nonisolated public func respond<Content>(
            within session: LanguageModelSession,
            to prompt: Prompt,
            generating type: Content.Type,
            includeSchemaInPrompt: Bool,
            options: GenerationOptions
        ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
            let fmSession = FoundationModels.LanguageModelSession(
                model: pccModel,
                tools: session.tools.toFoundationModels(),
                transcript: fmTranscriptDroppingDuplicatePrompt(session.transcript, prompt: prompt).toFoundationModels(
                    instructions: session.instructions,
                    toolDefinitions: session.tools
                        .filter(\.includesSchemaInInstructions)
                        .map { Transcript.ToolDefinition(tool: $0) }
                )
            )
            return try await fmRespond(
                makeSession: { fmSession },
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
            let fmSession = FoundationModels.LanguageModelSession(
                model: pccModel,
                tools: session.tools.toFoundationModels(),
                transcript: fmTranscriptDroppingDuplicatePrompt(session.transcript, prompt: prompt).toFoundationModels(
                    instructions: session.instructions,
                    toolDefinitions: session.tools
                        .filter(\.includesSchemaInInstructions)
                        .map { Transcript.ToolDefinition(tool: $0) }
                )
            )
            return fmStreamResponse(
                makeSession: { fmSession },
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
            let fmSession = FoundationModels.LanguageModelSession(
                model: pccModel,
                tools: session.tools.toFoundationModels(),
                instructions: session.instructions?.toFoundationModels()
            )
            return fmSession.logFeedbackAttachment(
                sentiment: sentiment?.toFoundationModels(),
                issues: issues.map { $0.toFoundationModels() },
                desiredOutput: nil
            )
        }
    }
#endif
