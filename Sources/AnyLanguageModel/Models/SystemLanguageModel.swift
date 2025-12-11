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

            let stream = AsyncThrowingStream<LanguageModelSession.ResponseStream<Content>.Snapshot, any Error> {
                @Sendable continuation in
                let task = Task {
                    // Bridge FoundationModels' stream into our ResponseStream snapshots
                    let fmStream: FoundationModels.LanguageModelSession.ResponseStream<String> =
                        fmSession.streamResponse(to: fmPrompt, options: fmOptions)

                    var accumulatedText = ""
                    do {
                        // Iterate FM stream of String snapshots
                        var lastLength = 0
                        for try await snapshot in fmStream {
                            var chunkText: String = snapshot.content

                            // We something get "null" from FoundationModels as a first temp result when streaming
                            // Some nil is probably converted to our String type when no data is available
                            if chunkText == "null" && accumulatedText == "" {
                                chunkText = ""
                            }

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

        private let wrappedTool: any Tool

        init(tool: any Tool) {
            self.wrappedTool = tool
            self.name = tool.name
            self.description = tool.description
            self.includesSchemaInInstructions = tool.includesSchemaInInstructions

            self.parameters = FoundationModels.GenerationSchema(tool.parameters)
        }

        public func call(arguments: FoundationModels.GeneratedContent) async throws -> Output {

            let output = try await wrappedTool.callFunction(arguments: arguments)
            // Since we can't call the tool's call method directly on an existential type,
            // we need to use makeOutputSegments which internally calls the tool
            // and then extract the result from the segments
            let result = output as! Output
            return result
        }
    }

    @available(macOS 26.0, *)
    extension FoundationModels.GenerationSchema {
        internal init(_ content: AnyLanguageModel.GenerationSchema) {
            let resolvedSchema = content.withResolvedRoot() ?? content

            let rawParameters = try? JSONValue(resolvedSchema)
            var schema: FoundationModels.GenerationSchema? = nil
            if rawParameters?.objectValue is [String: JSONValue] {
                if let data = try? JSONEncoder().encode(rawParameters) {
                    if let jsonSchema = try? JSONDecoder().decode(JSONSchema.self, from: data) {
                        if let dynamicSchema = SchemaConverter.convert(schema: jsonSchema) {
                            schema = try? FoundationModels.GenerationSchema(root: dynamicSchema, dependencies: [])
                        }
                    }
                }
            }
            if let schema = schema {
                self = schema
            } else {
                self = FoundationModels.GenerationSchema(
                    type: String.self,
                    properties: []
                )

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
    extension Tool {
        func callFunction(arguments: FoundationModels.GeneratedContent) async throws -> Output {

            let content = try GeneratedContent(arguments)
            return try await call(arguments: Self.Arguments(content))
        }
    }
    // MARK: - Errors
    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
    private enum SystemLanguageModelError: Error {
        case streamingFailed
    }

    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
    private
        class SchemaConverter
    {

        let jsonSchema: JSONSchema

        static func convert(schema: JSONSchema) -> FoundationModels.DynamicGenerationSchema? {
            let converter = SchemaConverter(schema: schema)
            return converter.convertedSchema()
        }

        init(schema: JSONSchema) {
            self.jsonSchema = schema
        }

        func convertedSchema() -> FoundationModels.DynamicGenerationSchema? {
            return schema(from: jsonSchema)
        }

        private func property(key: String, withSchema jsonSchema: JSONSchema, inSchema mainSchema: JSONSchema)
            -> FoundationModels.DynamicGenerationSchema.Property?
        {
            let required: [String] =
                if case .object(_, _, _, _, _, _, _, required: let required, _) = mainSchema {
                    required
                } else {
                    []
                }
            let isOptional = required.contains(key) == false
            let schema = self.schema(from: jsonSchema)

            switch jsonSchema {
            case .string(_, let description, _, _, _, _, _, _, _, _),
                .object(_, let description, _, _, _, _, _, _, _),
                .array(_, let description, _, _, _, _, _, _, _, _),
                .number(_, let description, _, _, _, _, _, _, _, _, _),
                .integer(_, let description, _, _, _, _, _, _, _, _, _),
                .boolean(_, let description, _):
                return .init(name: key, description: description, schema: schema, isOptional: isOptional)

            default:
                return .init(name: key, schema: schema, isOptional: isOptional)
            }
        }

        func schema(from jsonSchema: JSONSchema) -> FoundationModels.DynamicGenerationSchema {
            switch jsonSchema {
            case .object(_, description: let description, _, _, _, _, properties: let properties, _, _):

                let schemaProperties = properties.compactMap {
                    self.property(key: $0.0, withSchema: $0.1, inSchema: jsonSchema)
                }
                return .init(name: "", description: description, properties: schemaProperties)

            case .string(_, _, _, _, enum: let `enum`, const: let `const`, _, _, pattern: let pattern, _):

                var guides: [FoundationModels.GenerationGuide<String>] = []
                if let `enum`, let values = `enum` as? [String] {
                    guides.append(.anyOf(values))
                }
                if let `const`, let value = `const`.stringValue {
                    guides.append(.constant(value))
                }
                if let pattern, let regex = try? Regex(pattern) {
                    guides.append(.pattern(regex))
                }
                return .init(type: String.self, guides: guides)

            case .integer(
                _,
                _,
                _,
                _,
                enum: let `enum`,
                const: let `const`,
                minimum: let minimum,
                maximum: let maximum,
                _,
                _,
                _
            ):

                var guides: [FoundationModels.GenerationGuide<Int>] = []

                if let `enum` {

                    let enumsSchema = `enum`.compactMap { self.constSchema(object: $0) }
                    return FoundationModels.DynamicGenerationSchema(
                        name: "",
                        anyOf: enumsSchema
                    )

                } else {
                    if let min = minimum {
                        guides.append(.minimum(min))
                    }
                    if let max = maximum {
                        guides.append(.maximum(max))
                    }
                    if let `const`, let value = `const`.intValue {
                        guides.append(.range(value ... value))
                    }

                    return .init(type: Int.self, guides: guides)
                }

            case .number(
                _,
                _,
                _,
                _,
                enum: let `enum`,
                const: let `const`,
                minimum: let minimum,
                maximum: let maximum,
                _,
                _,
                _
            ):

                var guides: [FoundationModels.GenerationGuide<Double>] = []

                if let `enum` {

                    let enumsSchema = `enum`.compactMap { self.constSchema(object: $0) }
                    return .init(name: "", anyOf: enumsSchema)

                } else {
                    if let min = minimum {
                        guides.append(.minimum(min))
                    }
                    if let max = maximum {
                        guides.append(.maximum(max))
                    }
                    if let `const`, let value = `const`.doubleValue {
                        guides.append(.range(value ... value))
                    }
                }

                return .init(type: Double.self, guides: guides)

            case .boolean(_, _, _):
                return .init(type: Bool.self)

            case .anyOf(let schemas):
                return .init(name: "", anyOf: schemas.compactMap { self.schema(from: $0) })

            case .array(_, _, _, _, _, _, items: let items, minItems: let minItems, maxItems: let maxItems, _):

                // Note: const and enums are ignored for array properties
                let itemsSchema =
                    if let items {
                        schema(from: items)
                    } else {
                        FoundationModels.DynamicGenerationSchema(type: String.self)
                    }
                return .init(arrayOf: itemsSchema, minimumElements: minItems, maximumElements: maxItems)

            case .reference(let name):
                return .init(referenceTo: name)

            default:
                break
            }
            return .init(type: String.self)
        }

        // "country": { "const": "United States of America" }
        // not handling array, object, or bool here
        private func constSchema(object: JSONValue) -> FoundationModels.DynamicGenerationSchema? {
            switch object {
            case .int(let value):
                FoundationModels.DynamicGenerationSchema(type: Int.self, guides: [.range(value ... value)])
            case .double(let value):
                FoundationModels.DynamicGenerationSchema(type: Double.self, guides: [.range(value ... value)])
            case .string(let value):
                FoundationModels.DynamicGenerationSchema(type: String.self, guides: [.constant(value)])
            case .null:
                nil
            case .object(_), .bool(_), .array(_):
                // bool, object and array constant not supported by dynamic schema
                nil
            }
        }
    }

#endif
