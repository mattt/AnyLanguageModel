#if canImport(FoundationModels)
    import FoundationModels
    import Foundation
    import PartialJSONDecoder

    import JSONSchema

    /// A language model that uses Apple Intelligence.
    ///
    /// Use this model to generate text using on-device language models provided by Apple.
    /// This model runs entirely on-device and doesn't send data to external servers.
    ///
    /// ```swift
    /// let model = SystemLanguageModel()
    /// ```
    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
    public actor SystemLanguageModel: LanguageModel {
        /// The reason the model is unavailable.
        public typealias UnavailableReason = FoundationModels.SystemLanguageModel.Availability.UnavailableReason

        let systemModel: FoundationModels.SystemLanguageModel

        /// The default system language model.
        public static var `default`: SystemLanguageModel {
            SystemLanguageModel()
        }

        /// Creates the default system language model.
        public init() {
            self.systemModel = FoundationModels.SystemLanguageModel.default
        }

        /// Creates a system language model for a specific use case.
        ///
        /// - Parameters:
        ///   - useCase: The intended use case for generation.
        ///   - guardrails: Safety guardrails to apply during generation.
        public init(
            useCase: FoundationModels.SystemLanguageModel.UseCase = .general,
            guardrails: FoundationModels.SystemLanguageModel.Guardrails = FoundationModels.SystemLanguageModel
                .Guardrails.default
        ) {
            self.systemModel = FoundationModels.SystemLanguageModel(useCase: useCase, guardrails: guardrails)
        }

        /// Creates a system language model with a custom adapter.
        ///
        /// - Parameters:
        ///   - adapter: The adapter to use with the base model.
        ///   - guardrails: Safety guardrails to apply during generation.
        public init(
            adapter: FoundationModels.SystemLanguageModel.Adapter,
            guardrails: FoundationModels.SystemLanguageModel.Guardrails = .default
        ) {
            self.systemModel = FoundationModels.SystemLanguageModel(adapter: adapter, guardrails: guardrails)
        }

        /// The availability status for the system language model.
        nonisolated public var availability: Availability<UnavailableReason> {
            switch systemModel.availability {
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

        nonisolated public func streamResponse<Content>(
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
            let fmStream: FoundationModels.LanguageModelSession.ResponseStream<String> =
                fmSession.streamResponse(to: fmPrompt, options: fmOptions)
            let fmBox = UnsafeSendableBox(value: fmStream)

            let stream = AsyncThrowingStream<LanguageModelSession.ResponseStream<Content>.Snapshot, any Error> {
                @Sendable continuation in
                let task = Task {
                    var accumulatedText = ""
                    do {
                        // Iterate FM stream of String snapshots
                        var lastLength = 0
                        for try await snapshot in fmBox.value {
                            let chunkText: String = snapshot.content

                            if chunkText.count >= lastLength, chunkText.hasPrefix(accumulatedText) {
                                // Cumulative; compute delta via previous length
                                let startIdx = chunkText.index(chunkText.startIndex, offsetBy: lastLength)
                                let delta = String(chunkText[startIdx...])
                                accumulatedText += delta
                                lastLength = chunkText.count
                            } else if chunkText.hasPrefix(accumulatedText) {
                                // Fallback cumulative detection
                                accumulatedText = chunkText
                                lastLength = chunkText.count
                            } else if accumulatedText.hasPrefix(chunkText) {
                                // In unlikely case of an unexpected shrink, reset to the full chunk
                                accumulatedText = chunkText
                                lastLength = chunkText.count
                            } else {
                                // Treat as delta and append
                                accumulatedText += chunkText
                                lastLength = accumulatedText.count
                            }

                            // Build raw content from plain text
                            let raw: GeneratedContent = GeneratedContent(accumulatedText)

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

        nonisolated public func logFeedbackAttachment(
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
public struct FoundationModelsToolBridge<ArgumentsType, OutputType, ToolType : FoundationModels.Tool<String, String> > : FoundationModels.Tool where ArgumentsType : FoundationModels.ConvertibleFromGeneratedContent, OutputType : FoundationModels.PromptRepresentable, ToolType: FoundationModels.Tool {
        public typealias Arguments = ArgumentsType
        public typealias Output = OutputType

        /// The name of the tool
        public var name: String {
            tool.name
        }

        /// The description of the tool
        public var description: String {
            tool.description
        }

        /// Whether to include the schema in instructions
        public var includesSchemaInInstructions: Bool {
            tool.includesSchemaInInstructions
        }

        public func call(arguments: ArgumentsType) async throws -> OutputType {
            let t = arguments as! any ConvertibleToGeneratedContent
            let str = t.generatedContent as! ToolType.Arguments
            let result = try await tool.call(arguments: str)
            return result as! OutputType
        }
        
        public var parameters: FoundationModels.GenerationSchema {
            do {
                let data = try JSONEncoder().encode(tool.parameters)
                if let parameters = try? JSONDecoder().decode(FoundationModels.GenerationSchema.self, from: data) {
                    return parameters
                }
            } catch {
                // Swallow encoding errors and fall back below
            }
            // Fallback schema when bridging fails
            return FoundationModels.GenerationSchema(type: String.self, properties: [])
        }

        let tool: ToolType

        public init(tool: ToolType) {
            self.tool = tool
        }
    }

    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
    extension Array where Element == (any Tool) {
        fileprivate func toFoundationModels() -> [any FoundationModels.Tool] {
            self.map { tool in
                return AnyToolWrapper(tool: tool)
            }
        }
    }
    
    // MARK: - Tool Wrapper
    /// A type-erased wrapper that bridges any Tool to FoundationModels.Tool
    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
private struct AnyToolWrapper: FoundationModels.Tool {
        public typealias Arguments = FoundationModels.GeneratedContent
        public typealias Output = String

        public let name: String
        public let description: String
        public let parameters: FoundationModels.GenerationSchema
        public let includesSchemaInInstructions: Bool
        
        private let wrappedTool: Tool

        init(tool: any Tool) {
            self.wrappedTool = tool
            self.name = tool.name
            self.description = tool.description
            self.includesSchemaInInstructions = tool.includesSchemaInInstructions


            var properties : [FoundationModels.GenerationSchema.Property] = []

            // Handle the case where the schema has a root reference
            let resolvedSchema = tool.parameters.withResolvedRoot() ?? tool.parameters
            let rawParameters = try? JSONValue(resolvedSchema)
            if let value = rawParameters?.objectValue {
                let requiredKeys = value["required"]?.arrayValue?.map { $0.stringValue } ?? []
                if let params = value["properties"] {
                    if let v = params.objectValue {
                        for (key, value) in v {
                            if let type = value.objectValue {
                                if let typeName = type["type"]?.stringValue {
                                    properties.append(.init(
                                        name: key,
                                        description: key,
                                        typeName: typeName,
                                        isOptional: requiredKeys.contains(key) == false
                                        ))
                                }
                            }
                        }
                    }
                }
            }

            self.parameters = FoundationModels.GenerationSchema(
                type: String.self,
                description: "Parameters for \(tool.name)",
                properties: properties
            )
        }
        
        public func call(arguments: FoundationModels.GeneratedContent) async throws -> Output {

            let output = try await wrappedTool.callFunction(arguments: arguments)
            // Since we can't call the tool's call method directly on an existential type,
            // we need to use makeOutputSegments which internally calls the tool
            // and then extract the result from the segments
            do {
                // Convert segments to a string representation
                let result = output as! Output

                return result
            } catch {
                // Return error information as string
                return "Tool call failed: \(error.localizedDescription)" as! Output
            }
        }
    }

    @available(macOS 26.0, *)
    extension FoundationModels.GeneratedContent {
        internal init(_ content: AnyLanguageModel.GeneratedContent) throws {
            try self.init(json: content.jsonString)
        }
    }

    @available(macOS 26.0, *)
    extension AnyLanguageModel.GeneratedContent {
        internal init(_ content: FoundationModels.GeneratedContent) throws {
            try self.init(json: content.jsonString)
        }
    }

    @available(macOS 26.0, *)
    extension Tool
    {
        func callFunction(arguments: FoundationModels.GeneratedContent) async throws -> Output {

            let content = try GeneratedContent(arguments)
            return try await call(arguments: Self.Arguments(content))
        }
    }

    @available(macOS 26.0, *)
    extension FoundationModels.GenerationSchema.Property {
        internal init(name: String,
                      description: String? = nil,
                      typeName: String,
                      isOptional: Bool = false
        ) {
            let isOptionalSuffix = isOptional ? "?" : ""
            let typeName = typeName + isOptionalSuffix
            switch typeName {
            case "string":
                self.init(name: name,
                          description: description,
                          type: String.self)

            case "string?":
                self.init(name: name,
                          description: description,
                          type: String?.self)

            case "int":
                self.init(name: name,
                          description: description,
                          type: Int.self)

            case "int?":
                self.init(name: name,
                          description: description,
                          type: Int?.self)

            case "bool":
                self.init(name: name,
                          description: description,
                          type: Bool.self)

            case "bool?":
                self.init(name: name,
                          description: description,
                          type: Bool?.self)


            default :
                self.init(name: name,
                          description: description,
                          type: String.self)
            }
        }
    }

    // MARK: - Errors

    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
    private enum SystemLanguageModelError: Error {
        case streamingFailed
    }


#endif


