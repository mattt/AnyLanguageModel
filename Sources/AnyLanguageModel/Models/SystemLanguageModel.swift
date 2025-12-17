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
                transcript: session.transcript.toFoundationModels(instructions: session.instructions)
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
                transcript: session.transcript.toFoundationModels(instructions: session.instructions)
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
            map { AnyToolWrapper($0) }
        }
    }

    /// A type-erased wrapper that bridges any `Tool` to `FoundationModels.Tool`.
    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
    private struct AnyToolWrapper: FoundationModels.Tool {
        typealias Arguments = FoundationModels.GeneratedContent
        typealias Output = String

        let name: String
        let description: String
        let parameters: FoundationModels.GenerationSchema
        let includesSchemaInInstructions: Bool

        private let wrappedTool: any Tool

        init(_ tool: any Tool) {
            self.wrappedTool = tool
            self.name = tool.name
            self.description = tool.description
            self.parameters = FoundationModels.GenerationSchema(tool.parameters)
            self.includesSchemaInInstructions = tool.includesSchemaInInstructions
        }

        func call(arguments: FoundationModels.GeneratedContent) async throws -> Output {
            let output = try await wrappedTool.callFunction(arguments: arguments)
            return output.promptRepresentation.description
        }
    }

    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
    extension FoundationModels.GenerationSchema {
        internal init(_ content: AnyLanguageModel.GenerationSchema) {
            let resolvedSchema = content.withResolvedRoot() ?? content

            let rawParameters = try? JSONValue(resolvedSchema)
            var schema: FoundationModels.GenerationSchema? = nil
            if rawParameters?.objectValue is [String: JSONValue] {
                if let data = try? JSONEncoder().encode(rawParameters) {
                    if let jsonSchema = try? JSONDecoder().decode(JSONSchema.self, from: data) {
                        let dynamicSchema = convertToDynamicSchema(jsonSchema)
                        schema = try? FoundationModels.GenerationSchema(root: dynamicSchema, dependencies: [])
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

    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
    extension FoundationModels.GeneratedContent {
        internal init(_ content: AnyLanguageModel.GeneratedContent) throws {
            try self.init(json: content.jsonString)
        }
    }

    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
    extension AnyLanguageModel.GeneratedContent {
        internal init(_ content: FoundationModels.GeneratedContent) throws {
            try self.init(json: content.jsonString)
        }
    }

    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
    extension Tool {
        fileprivate func callFunction(arguments: FoundationModels.GeneratedContent) async throws
            -> any PromptRepresentable
        {
            let content = try GeneratedContent(arguments)
            return try await call(arguments: Self.Arguments(content))
        }
    }

    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
    func convertToDynamicSchema(_ jsonSchema: JSONSchema) -> FoundationModels.DynamicGenerationSchema {
        switch jsonSchema {
        case .object(_, _, _, _, _, _, properties: let properties, required: let required, _):
            let schemaProperties = properties.compactMap { key, value in
                convertToProperty(key: key, schema: value, required: required)
            }
            return .init(name: "", description: jsonSchema.description, properties: schemaProperties)

        case .string(_, _, _, _, _, _, _, _, pattern: let pattern, _):
            var guides: [FoundationModels.GenerationGuide<String>] = []
            if let values = jsonSchema.enum?.compactMap(\.stringValue), !values.isEmpty {
                guides.append(.anyOf(values))
            }
            if let value = jsonSchema.const?.stringValue {
                guides.append(.constant(value))
            }
            if let pattern, let regex = try? Regex(pattern) {
                guides.append(.pattern(regex))
            }
            return .init(type: String.self, guides: guides)

        case .integer(_, _, _, _, _, _, minimum: let minimum, maximum: let maximum, _, _, _):
            if let enumValues = jsonSchema.enum {
                let enumsSchema = enumValues.compactMap { convertConstToSchema($0) }
                return .init(name: "", anyOf: enumsSchema)
            }

            var guides: [FoundationModels.GenerationGuide<Int>] = []
            if let min = minimum {
                guides.append(.minimum(min))
            }
            if let max = maximum {
                guides.append(.maximum(max))
            }
            if let value = jsonSchema.const?.intValue {
                guides.append(.range(value ... value))
            }
            return .init(type: Int.self, guides: guides)

        case .number(_, _, _, _, _, _, minimum: let minimum, maximum: let maximum, _, _, _):
            if let enumValues = jsonSchema.enum {
                let enumsSchema = enumValues.compactMap { convertConstToSchema($0) }
                return .init(name: "", anyOf: enumsSchema)
            }

            var guides: [FoundationModels.GenerationGuide<Double>] = []
            if let min = minimum {
                guides.append(.minimum(min))
            }
            if let max = maximum {
                guides.append(.maximum(max))
            }
            if let value = jsonSchema.const?.doubleValue {
                guides.append(.range(value ... value))
            }
            return .init(type: Double.self, guides: guides)

        case .boolean:
            return .init(type: Bool.self)

        case .anyOf(let schemas):
            return .init(name: "", anyOf: schemas.map { convertToDynamicSchema($0) })

        case .array(_, _, _, _, _, _, items: let items, minItems: let minItems, maxItems: let maxItems, _):
            let itemsSchema =
                items.map { convertToDynamicSchema($0) }
                ?? FoundationModels.DynamicGenerationSchema(type: String.self)
            return .init(arrayOf: itemsSchema, minimumElements: minItems, maximumElements: maxItems)

        case .reference(let name):
            return .init(referenceTo: name)

        case .allOf, .oneOf, .not, .null, .empty, .any:
            return .init(type: String.self)
        }
    }

    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
    func convertToProperty(
        key: String,
        schema: JSONSchema,
        required: [String]
    ) -> FoundationModels.DynamicGenerationSchema.Property {
        .init(
            name: key,
            description: schema.description,
            schema: convertToDynamicSchema(schema),
            isOptional: !required.contains(key)
        )
    }

    /// Converts a JSON constant value to a DynamicGenerationSchema.
    /// Only handles scalar types (int, double, string); returns nil for null, object, bool, and array.
    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
    func convertConstToSchema(_ value: JSONValue) -> FoundationModels.DynamicGenerationSchema? {
        switch value {
        case .int(let intValue):
            .init(type: Int.self, guides: [.range(intValue ... intValue)])
        case .double(let doubleValue):
            .init(type: Double.self, guides: [.range(doubleValue ... doubleValue)])
        case .string(let stringValue):
            .init(type: String.self, guides: [.constant(stringValue)])
        case .null, .object, .bool, .array:
            nil
        }
    }

    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
    extension Transcript {
        fileprivate func toFoundationModels(instructions: AnyLanguageModel.Instructions?) -> FoundationModels.Transcript
        {
            var fmEntries: [FoundationModels.Transcript.Entry] = []

            // Add instructions entry if provided and not already in transcript
            if let instructions = instructions {
                let hasInstructions =
                    self.first.map { entry in
                        if case .instructions = entry { return true } else { return false }
                    } ?? false

                if !hasInstructions {
                    let fmInstructions = FoundationModels.Transcript.Instructions(
                        segments: [.text(.init(content: instructions.description))],
                        toolDefinitions: []
                    )
                    fmEntries.append(.instructions(fmInstructions))
                }
            }

            // Convert each entry
            for entry in self {
                switch entry {
                case .instructions(let instr):
                    let fmSegments = instr.segments.toFoundationModels()
                    let fmToolDefinitions = instr.toolDefinitions.toFoundationModels()
                    let fmInstructions = FoundationModels.Transcript.Instructions(
                        segments: fmSegments,
                        toolDefinitions: fmToolDefinitions
                    )
                    fmEntries.append(.instructions(fmInstructions))

                case .prompt(let prompt):
                    let fmSegments = prompt.segments.toFoundationModels()
                    let fmPrompt = FoundationModels.Transcript.Prompt(
                        segments: fmSegments
                    )
                    fmEntries.append(.prompt(fmPrompt))

                case .response(let response):
                    let fmSegments = response.segments.toFoundationModels()
                    let fmResponse = FoundationModels.Transcript.Response(
                        assetIDs: response.assetIDs,
                        segments: fmSegments
                    )
                    fmEntries.append(.response(fmResponse))

                case .toolCalls(let toolCalls):
                    let fmCalls = toolCalls.compactMap { call -> FoundationModels.Transcript.ToolCall? in
                        guard let fmArguments = try? FoundationModels.GeneratedContent(call.arguments) else {
                            return nil
                        }
                        return FoundationModels.Transcript.ToolCall(
                            id: call.id,
                            toolName: call.toolName,
                            arguments: fmArguments
                        )
                    }
                    let fmToolCalls = FoundationModels.Transcript.ToolCalls(id: toolCalls.id, fmCalls)
                    fmEntries.append(.toolCalls(fmToolCalls))

                case .toolOutput(let toolOutput):
                    let fmSegments = toolOutput.segments.toFoundationModels()
                    let fmToolOutput = FoundationModels.Transcript.ToolOutput(
                        id: toolOutput.id,
                        toolName: toolOutput.toolName,
                        segments: fmSegments
                    )
                    fmEntries.append(.toolOutput(fmToolOutput))
                }
            }

            return FoundationModels.Transcript(entries: fmEntries)
        }
    }

    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
    extension Array where Element == Transcript.Segment {
        fileprivate func toFoundationModels() -> [FoundationModels.Transcript.Segment] {
            compactMap { segment -> FoundationModels.Transcript.Segment? in
                switch segment {
                case .text(let textSegment):
                    return .text(.init(id: textSegment.id, content: textSegment.content))
                case .structure(let structuredSegment):
                    guard let fmContent = try? FoundationModels.GeneratedContent(structuredSegment.content) else {
                        return nil
                    }
                    return .structure(
                        .init(
                            id: structuredSegment.id,
                            source: structuredSegment.source,
                            content: fmContent
                        )
                    )
                case .image:
                    // FoundationModels Transcript does not support image segments
                    return nil
                }
            }
        }
    }

    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
    extension Array where Element == Transcript.ToolDefinition {
        fileprivate func toFoundationModels() -> [FoundationModels.Transcript.ToolDefinition] {
            map { toolDef in
                FoundationModels.Transcript.ToolDefinition(
                    name: toolDef.name,
                    description: toolDef.description,
                    parameters: FoundationModels.GenerationSchema(toolDef.parameters)
                )
            }
        }
    }
#endif
