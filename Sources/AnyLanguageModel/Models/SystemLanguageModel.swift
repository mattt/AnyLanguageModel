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
            var schema : FoundationModels.GenerationSchema? = nil
            if let parameters = rawParameters?.objectValue as? [String: JSONValue] {
                let convertor = ValueSchemaConverter(root: parameters)
                if let dynamicSchema = convertor.schema() {
                    schema = try? FoundationModels.GenerationSchema(root: dynamicSchema, dependencies: [])
                }
            }
            if let schema = schema {
                self.parameters = schema
            } else {
                self.parameters = FoundationModels.GenerationSchema(
                    type: String.self,
                    description: "tool parameters",
                    properties: [])

            }
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
    extension FoundationModels.GenerationSchema {
        internal init(_ content: AnyLanguageModel.GenerationSchema) throws {
            let resolvedContent = content
            do {
                let data = try JSONEncoder().encode(resolvedContent)
                self = try JSONDecoder().decode(FoundationModels.GenerationSchema.self, from: data)
            }
            catch {
                let data = try JSONEncoder().encode(content)
                self = try JSONDecoder().decode(FoundationModels.GenerationSchema.self, from: data)
            }
        }
    }

    @available(macOS 26.0, *)
    extension AnyLanguageModel.GenerationSchema {
        internal init(_ content: FoundationModels.GenerationSchema) throws {
            let data = try JSONEncoder().encode(content)
            self = try JSONDecoder().decode(AnyLanguageModel.GenerationSchema.self, from: data)
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
    // MARK: - Errors
    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
    private enum SystemLanguageModelError: Error {
        case streamingFailed
    }

// Inspired by :  https://github.com/0Itsuki0/Swift_FoundationModelsWithMCP/blob/main/FoundationModelWithMCP/ValueSchemaConvertor.swift
    // MARK: - JSON Schema Keys
    private enum JSONSchemaKey: String, CustomStringConvertible {
        case type = "type"
        case properties = "properties"
        case items = "items"
        case required = "required"
        case description = "description"
        case title = "title"
        case enumType = "enum"
        case constType = "const"
        case anyOf = "anyOf"

        // Types
        case null = "null"
        case boolean = "boolean"
        case number = "number"
        case integer = "integer"
        case array = "array"
        case string = "string"
        case object = "object"

        // Array constraints
        case minItems = "minItems"
        case maxItems = "maxItems"

        var description: String {
            return self.rawValue
        }
    }


    // MARK: ValueSchemaConvertor
    // converting JSON Schema (JSONValue) to Dynamic Generation Schema
    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
    class ValueSchemaConverter {

        let root: JSONValue
        var decodingStack: Set<String> = []

        init(root: [String: JSONValue]) {
            self.root = JSONValue.object(root)
        }

        func schema() -> FoundationModels.DynamicGenerationSchema? {
            return schema(from: root.objectValue!)
        }

        private func resolvedValue(_ value: JSONValue) -> JSONValue {
            switch value {

            case .null, .bool, .int, .double, .string:
                return value

            case .array(let array):
                return .array(array.compactMap { resolvedValue($0) })

            case .object(let object):
                if let ref = object["$ref"]?.stringValue {
                    if decodingStack.contains(ref) {
                        return value // Circular references
                    }
                    var resolvedRef = value
                    decodingStack.insert(ref)
                    var success = false
                    // Support only of #/path/to/object" references for now
                    if ref.hasPrefix("#") {
                        let components = ref.split(separator: "/")
                        var result = root
                        for key in components {
                            if key == "#" {
                                continue
                            }
                            if let object = result.objectValue {
                                if let val = object[String(key)] {
                                    success = true
                                    result = val
                                } else {
                                    success = false
                                    result = value
                                    break
                                }
                            }
                        }
                        resolvedRef = success ? resolvedValue(result) : result
                    }
                    decodingStack.remove(ref)
                    return resolvedRef
                } else {
                    return .object(object.compactMapValues { resolvedValue($0) })

                }
            }
            return value
        }
        private func propertyFrom(value: [String: JSONValue], name: String, description: String?, typeString: String?, required: Bool) -> FoundationModels.DynamicGenerationSchema.Property? {
            let propertyName = name
            let propertyDescription = description
            var propertyTypeString = typeString
            var isRequired = required

            if typeString == nil {
                // Check if it's a "anyOf <types>"
                if let typeValue = value[JSONSchemaKey.anyOf.rawValue]?.arrayValue {
                    let typeArray = typeValue.compactMap {$0.objectValue?[JSONSchemaKey.type.rawValue]?.stringValue}
                    // We don't actually support than for now - just return the first non-null one
                    if typeArray.contains(JSONSchemaKey.null.rawValue) {
                        isRequired = false
                    }
                    propertyTypeString = typeArray.first { $0 != JSONSchemaKey.null.rawValue }
                }
            }
            switch propertyTypeString {
            case JSONSchemaKey.null.rawValue:
                return nil

            case JSONSchemaKey.boolean.rawValue:
                let schema = FoundationModels.DynamicGenerationSchema(type: Bool.self)
                return property(name: propertyName, description: propertyDescription, schema: schema, required: isRequired)

            case JSONSchemaKey.object.rawValue:
                let schema = schema(from: value)
                return property(name: propertyName, description: propertyDescription, schema: schema, required: isRequired)

            case JSONSchemaKey.array.rawValue:
                let arrayItems = getArrayItems(object: value)
                let (min, max) = getArrayMinMax(object: value)

                let schema = schema(from: arrayItems)
                let arraySchema = FoundationModels.DynamicGenerationSchema(
                    arrayOf: schema,
                    minimumElements: min,
                    maximumElements: max
                )
                return property(name: propertyName, description: propertyDescription, schema: arraySchema, required: isRequired)

            case JSONSchemaKey.number.rawValue:
                let result = applyEnumAndConstConstraints(for: value, name: propertyName, description: propertyDescription, defaultSchema: FoundationModels.DynamicGenerationSchema(type: Double.self), required: isRequired)
                return property(name: propertyName, description: propertyDescription, schema: result.schema, required: result.isRequired)

            case JSONSchemaKey.integer.rawValue:
                let result = applyEnumAndConstConstraints(for: value, name: propertyName, description: propertyDescription, defaultSchema: FoundationModels.DynamicGenerationSchema(type: Int.self), required: isRequired)
                return property(name: propertyName, description: propertyDescription, schema: result.schema, required: result.isRequired)

            case JSONSchemaKey.string.rawValue:
                let result = applyEnumAndConstConstraints(for: value, name: propertyName, description: propertyDescription, defaultSchema: FoundationModels.DynamicGenerationSchema(type: String.self), required: isRequired)
                return property(name: propertyName, description: propertyDescription, schema: result.schema, required: result.isRequired)

            case JSONSchemaKey.anyOf.rawValue:
                if let value = value[JSONSchemaKey.anyOf.rawValue]?.arrayValue {
                    let typeArray = value.compactMap { $0.objectValue }
                    let schemasArray: [FoundationModels.DynamicGenerationSchema] = typeArray.compactMap { self.schema(from: $0) }
                    let schema = FoundationModels.DynamicGenerationSchema(name: propertyName, description: description, anyOf: schemasArray)
                    return property(name: name, description: description, schema: schema, required: isRequired)
                }

            default:
                // const or enum without type
                if let enumValue = value[JSONSchemaKey.enumType.rawValue] {
                    let result = enumSchema(name: propertyName, description: description, object: enumValue)
                    return property(name: propertyName, description: propertyDescription, schema: result.schema, required: result.isRequired ?? isRequired)
                }

                if let constValue = value[JSONSchemaKey.constType.rawValue], let schema = constSchema(object: constValue) {
                    return property(name: propertyName, description: propertyDescription, schema: schema, required: isRequired)
                }
            }

            return nil
        }

        func schema(from object: [String: JSONValue]) -> FoundationModels.DynamicGenerationSchema {
            let title = getTitle(object: object) ?? UUID().uuidString
            let description = getDescription(object: object)

            if let type = getPropertyTypeString(object: object) {
                switch type {
                case JSONSchemaKey.boolean.rawValue:
                    return FoundationModels.DynamicGenerationSchema(type: Bool.self)

                case JSONSchemaKey.array.rawValue:
                    let arrayItems = getArrayItems(object: object)
                    let (min, max) = getArrayMinMax(object: object)
                    let itemSchema = schema(from: arrayItems)

                    return FoundationModels.DynamicGenerationSchema(
                        arrayOf: itemSchema,
                        minimumElements: min,
                        maximumElements: max
                    )

                case JSONSchemaKey.number.rawValue:
                    return applyEnumAndConstConstraints(for: object, name: title, description: description, defaultSchema: FoundationModels.DynamicGenerationSchema(type: Double.self), required: true).schema

                case JSONSchemaKey.integer.rawValue:
                    return applyEnumAndConstConstraints(for: object, name: title, description: description, defaultSchema: FoundationModels.DynamicGenerationSchema(type: Int.self), required: true).schema

                case JSONSchemaKey.string.rawValue:
                    return applyEnumAndConstConstraints(for: object, name: title, description: description, defaultSchema: FoundationModels.DynamicGenerationSchema(type: String.self), required: true).schema

                case JSONSchemaKey.anyOf.rawValue:
                    if let value = object[JSONSchemaKey.anyOf.rawValue]?.arrayValue as? [[String: JSONValue]] {
                        let schemasArray: [FoundationModels.DynamicGenerationSchema] = value.compactMap { schema(from: $0) }
                        return FoundationModels.DynamicGenerationSchema(name: title, description: description, anyOf: schemasArray)
                    }

                case JSONSchemaKey.object.rawValue:
                    return objectSchema(from: object, title: title, description: description)

                default:
                    // Unknown type, fall through to handle as object or typeless enum/const
                    break
                }
            }

            // Handle schemas without a 'type' property, or with an unhandled type
            if let value = object[JSONSchemaKey.enumType.rawValue] {
                return enumSchema(name: title, description: description, object: value).schema
            }

            if let value = object[JSONSchemaKey.constType.rawValue], let result = constSchema(object: value) {
                return result
            }

            // Fallback for objects without an explicit "type": "object"
            return objectSchema(from: object, title: title, description: description)
        }

        private func objectSchema(from object: [String: JSONValue], title: String, description: String?) -> FoundationModels.DynamicGenerationSchema {
            let requiredFields = getRequiredFields(object: object)
            let properties = getProperties(object: object)

            let schemaProperties: [FoundationModels.DynamicGenerationSchema.Property] = properties.compactMap { (key, value) in
                if let value = resolvedValue(JSONValue.object(value)).objectValue {
                    let propertyName: String = key
                    let propertyDescription = getDescription(object: value)
                    let propertyTypeString = getPropertyTypeString(object: value)
                    let required = requiredFields.contains(key)

                    return propertyFrom(value: value, name: propertyName, description: propertyDescription, typeString: propertyTypeString, required: required)
                } else {
                    return nil
                }
            }

            return FoundationModels.DynamicGenerationSchema(
                name: title,
                description: description,
                properties: schemaProperties
            )
        }

        private func applyEnumAndConstConstraints(for value: [String: JSONValue], name: String, description: String?, defaultSchema: FoundationModels.DynamicGenerationSchema, required: Bool) -> (schema: FoundationModels.DynamicGenerationSchema, isRequired: Bool) {
            var schema = defaultSchema
            var isRequired = required

            if let enumValue = value[JSONSchemaKey.enumType.rawValue] {
                let result = enumSchema(name: name, description: description, object: enumValue)
                schema = result.schema
                if let r = result.isRequired {
                    isRequired = r
                }
            } else if let constValue = value[JSONSchemaKey.constType.rawValue], let result = constSchema(object: constValue) {
                schema = result
            }

            return (schema, isRequired)
        }

        private func property(name: String, description: String?, schema: FoundationModels.DynamicGenerationSchema, required: Bool)  -> FoundationModels.DynamicGenerationSchema.Property {
            return FoundationModels.DynamicGenerationSchema.Property(
                name: name,
                description: description,
                schema: schema,
                isOptional: required == false
            )
        }

        // "color": { "enum": ["red", "amber", "green", null, 42] }
        private func enumSchema(name: String, description: String?, object: JSONValue) -> (schema: FoundationModels.DynamicGenerationSchema, isRequired: Bool?) {
            guard let array = object.arrayValue else {
                let schema = FoundationModels.DynamicGenerationSchema(name: name, description: description, anyOf: [] as [String])
                return (schema: schema, isRequired: nil)
            }

            let strings = array.compactMap { $0.stringValue }

            if strings.count == array.count {
                let stringSchema = FoundationModels.DynamicGenerationSchema(name: name, description: description, anyOf: strings)
                return (schema: stringSchema, isRequired: nil)
            }

            let required = !array.contains { $0.isNull }

            let nonStringValues = array.filter { $0.stringValue == nil }

            var schemas: [FoundationModels.DynamicGenerationSchema] = []
            if !strings.isEmpty {
                schemas.append(FoundationModels.DynamicGenerationSchema(name: name, description: description, anyOf: strings))
            }

            for value in nonStringValues {
                if let intValue = value.intValue {
                    schemas.append(constantIntSchema(value: intValue))
                } else if let doubleValue = value.doubleValue {
                    schemas.append(constantDoubleSchema(value: doubleValue))
                } else if value.boolValue != nil {
                    schemas.append(FoundationModels.DynamicGenerationSchema(type: Bool.self))
                }
            }

            let schema = FoundationModels.DynamicGenerationSchema(name: name, description: description, anyOf: schemas)
            return (schema: schema, isRequired: required)
        }


        // "country": { "const": "United States of America" }
        // not handling array, object, or bool here
        private func constSchema(object: JSONValue) -> FoundationModels.DynamicGenerationSchema? {
            if let intValue = object.intValue {
                return constantIntSchema(value: intValue)
            }
            if let doubleValue = object.doubleValue {
                return constantDoubleSchema(value: doubleValue)
            }
            if let stringValue = object.stringValue {
                return constantStringSchema(value: stringValue)
            }
            if let _ = object.arrayValue {
                // array constant not supported by dynamic schema
                return nil
            }

            if let _ = object.objectValue {
                // object constant not supported by dynamic schema
                return nil
            }
            return nil
        }

        private func constantIntSchema(value: Int) -> FoundationModels.DynamicGenerationSchema {
            FoundationModels.DynamicGenerationSchema(type: Int.self, guides: [.range(value...value)])
        }

        private func constantDoubleSchema(value: Double) -> FoundationModels.DynamicGenerationSchema {
            FoundationModels.DynamicGenerationSchema(type: Double.self, guides: [.range(value...value)])
        }

        private func constantStringSchema(value: String) -> FoundationModels.DynamicGenerationSchema {
            FoundationModels.DynamicGenerationSchema(type: String.self, guides: [.constant(value)])
        }

        private func getRequiredFields(object: [String: JSONValue]) -> [String] {
            guard let array = object[JSONSchemaKey.required.rawValue]?.arrayValue else {
                return []
            }
            return array.compactMap { $0.stringValue }
        }

        // minItems and maxItems
        private func getArrayMinMax(object: [String: JSONValue]) -> (Int?, Int?) {
            let minInt = object[JSONSchemaKey.minItems.rawValue]?.intValue
            let maxInt = object[JSONSchemaKey.maxItems.rawValue]?.intValue
            return (minInt, maxInt)
        }

        private func getArrayItems(object: [String: JSONValue]) -> [String: JSONValue] {
            guard let items = object[JSONSchemaKey.items.rawValue]?.objectValue else {
                return [:]
            }
            return items
        }

        private func getProperties(object: [String: JSONValue]) -> [String: [String: JSONValue]] {
            guard let propertyObject = object[JSONSchemaKey.properties.rawValue]?.objectValue else {
                return [:]
            }
            return propertyObject.compactMapValues { $0.objectValue }
        }

        private func getDescription(object: [String: JSONValue]) -> String? {
            return object[JSONSchemaKey.description.rawValue]?.stringValue
        }

        private func getRequired(object: [String: JSONValue]) -> [String]? {
            guard let required = object[JSONSchemaKey.required.rawValue]?.arrayValue as? [String] else {
                return nil
            }
            return required
        }

        private func getTitle(object: [String: JSONValue]) -> String? {
            return object[JSONSchemaKey.title.rawValue]?.stringValue
        }

        private func getPropertyTypeString(object: [String: JSONValue]) -> String? {
            return object[JSONSchemaKey.type.rawValue]?.stringValue
        }
    }

#endif


