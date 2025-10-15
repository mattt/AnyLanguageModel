#if canImport(FoundationModels)
    import FoundationModels
    import Foundation
    import PartialJSONDecoder

    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
    public struct SystemLanguageModel {
        let systemModel: FoundationModels.SystemLanguageModel

        /// Creates the default system language model.
        public init() {
            self.systemModel = FoundationModels.SystemLanguageModel.default
        }

        /// Creates a system language model for a specific use case.
        public init(
            useCase: FoundationModels.SystemLanguageModel.UseCase = .general,
            guardrails: FoundationModels.SystemLanguageModel.Guardrails = FoundationModels.SystemLanguageModel
                .Guardrails.default
        ) {
            self.systemModel = FoundationModels.SystemLanguageModel(useCase: useCase, guardrails: guardrails)
        }

        /// Creates the base version of the model with an adapter.
        public init(
            adapter: FoundationModels.SystemLanguageModel.Adapter,
            guardrails: FoundationModels.SystemLanguageModel.Guardrails = .default
        ) {
            self.systemModel = FoundationModels.SystemLanguageModel(adapter: adapter, guardrails: guardrails)
        }
    }

    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
    extension SystemLanguageModel: LanguageModel {
        public func logFeedbackAttachment(
            within session: LanguageModelSession,
            sentiment: LanguageModelFeedback.Sentiment?,
            issues: [LanguageModelFeedback.Issue],
            desiredOutput: Transcript.Entry?
        ) -> Data {
            let fmSession = FoundationModels.LanguageModelSession(
                model: systemModel,
                tools: session.tools.toFoundationModels(),
                instructions: session.instructions?.toFoundationModels()
            )

            let fmSentiment = sentiment?.toFoundationModels()
            let fmIssues = issues.map { $0.toFoundationModels() }
            let fmDesiredOutput: FoundationModels.Transcript.Entry? = nil

            return fmSession.logFeedbackAttachment(
                sentiment: fmSentiment,
                issues: fmIssues,
                desiredOutput: fmDesiredOutput
            )
        }

        public func respond<Content>(
            within session: LanguageModelSession,
            to prompt: Prompt,
            generating type: Content.Type,
            includeSchemaInPrompt: Bool,
            options: GenerationOptions
        ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
            let fmPrompt = prompt.toFoundationModels()
            let fmOptions = options.toFoundationModels()

            let fmSession = FoundationModels.LanguageModelSession(
                model: systemModel,
                tools: session.tools.toFoundationModels(),
                instructions: session.instructions?.toFoundationModels()
            )

            let fmResponse = try await fmSession.respond(to: fmPrompt, options: fmOptions)
            let generatedContent = GeneratedContent(fmResponse.content)

            if type == String.self {
                return LanguageModelSession.Response(
                    content: fmResponse.content as! Content,
                    rawContent: generatedContent,
                    transcriptEntries: []
                )
            } else {
                // For non-String types, try to create an instance from the generated content
                let content = try type.init(generatedContent)

                return LanguageModelSession.Response(
                    content: content,
                    rawContent: generatedContent,
                    transcriptEntries: []
                )
            }
        }

        public func streamResponse<Content>(
            within session: LanguageModelSession,
            to prompt: Prompt,
            generating type: Content.Type,
            includeSchemaInPrompt: Bool,
            options: GenerationOptions
        ) -> sending LanguageModelSession.ResponseStream<Content> where Content: Generable {
            let fmPrompt = prompt.toFoundationModels()
            let fmOptions = options.toFoundationModels()

            let fmSession = FoundationModels.LanguageModelSession(
                model: systemModel,
                tools: session.tools.toFoundationModels(),
                instructions: session.instructions?.toFoundationModels()
            )

            // Bridge FoundationModels' stream into our ResponseStream snapshots
            let fmStream = fmSession.streamResponse(to: fmPrompt, options: fmOptions)
            let fmBox = UnsafeSendableBox(value: fmStream)

            let stream = AsyncThrowingStream<LanguageModelSession.ResponseStream<Content>.Snapshot, any Error> {
                @Sendable continuation in
                let task = Task {
                    var accumulatedText = ""
                    do {
                        // Iterate FM stream; assume it yields textual chunks or objects convertible to String
                        for try await chunk in fmBox.value {
                            let chunkText = String(describing: chunk)

                            accumulatedText += chunkText

                            // Build raw content from partial JSON or plain text
                            let raw: GeneratedContent =
                                (try? GeneratedContent(json: accumulatedText)) ?? GeneratedContent(accumulatedText)

                            // Materialize Content when possible
                            let snapshotContent: Content.PartiallyGenerated = {
                                if type == String.self {
                                    return (accumulatedText as! Content).asPartiallyGenerated()
                                }
                                if let value = try? type.init(raw) {
                                    return value.asPartiallyGenerated()
                                }
                                // As a last resort, expose raw as partially generated if compatible
                                return (try? type.init(GeneratedContent(accumulatedText)))?.asPartiallyGenerated()
                                    ?? ("" as! Content).asPartiallyGenerated()
                            }()

                            continuation.yield(.init(content: snapshotContent, rawContent: raw))
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

    // MARK: - Helpers

    // Minimal box to allow capturing non-Sendable values in @Sendable closures safely.
    private struct UnsafeSendableBox<T>: @unchecked Sendable { let value: T }

    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
    extension Prompt {
        fileprivate func toFoundationModels() -> FoundationModels.Prompt {
            FoundationModels.Prompt(self.description)
        }
    }

    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
    extension Instructions {
        fileprivate func toFoundationModels() -> FoundationModels.Instructions {
            FoundationModels.Instructions(self.description)
        }
    }

    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
    extension GenerationOptions {
        fileprivate func toFoundationModels() -> FoundationModels.GenerationOptions {
            var options = FoundationModels.GenerationOptions()

            if let temperature = self.temperature {
                options.temperature = temperature
            }

            // Note: FoundationModels.GenerationOptions may not have all properties
            // Only set those that are available

            return options
        }
    }

    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
    extension LanguageModelFeedback.Sentiment {
        fileprivate func toFoundationModels() -> FoundationModels.LanguageModelFeedback.Sentiment {
            switch self {
            case .positive: .positive
            case .negative: .negative
            case .neutral: .neutral
            }
        }
    }

    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
    extension LanguageModelFeedback.Issue {
        fileprivate func toFoundationModels() -> FoundationModels.LanguageModelFeedback.Issue {
            FoundationModels.LanguageModelFeedback.Issue(
                category: self.category.toFoundationModels(),
                explanation: self.explanation
            )
        }
    }

    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
    extension LanguageModelFeedback.Issue.Category {
        fileprivate func toFoundationModels() -> FoundationModels.LanguageModelFeedback.Issue.Category {
            switch self {
            case .unhelpful: .unhelpful
            case .tooVerbose: .tooVerbose
            case .didNotFollowInstructions: .didNotFollowInstructions
            case .incorrect: .incorrect
            case .stereotypeOrBias: .stereotypeOrBias
            case .suggestiveOrSexual: .suggestiveOrSexual
            case .vulgarOrOffensive: .vulgarOrOffensive
            case .triggeredGuardrailUnexpectedly: .triggeredGuardrailUnexpectedly
            }
        }
    }

    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
    extension Array where Element == (any Tool) {
        fileprivate func toFoundationModels() -> [any FoundationModels.Tool] {
            return []
        }
    }

    // MARK: - Errors

    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
    private enum SystemLanguageModelError: Error {
        case streamingFailed
    }

#endif
