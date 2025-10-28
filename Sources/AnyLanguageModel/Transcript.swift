import struct Foundation.UUID

/// A type that represents a conversation history between a user and a language model.
public struct Transcript: Sendable, Equatable, Codable {
    var entries: [Entry]

    /// Creates a transcript.
    ///
    /// - Parameters:
    ///   - entries: An array of entries to seed the transcript.
    public init(entries: some Sequence<Entry> = []) {
        self.entries = Array(entries)
    }

    /// An entry in a transcript.
    public enum Entry: Sendable, Identifiable, Equatable, Codable {
        /// Instructions, typically provided by you, the developer.
        case instructions(Instructions)

        /// A prompt, typically sourced from an end user.
        case prompt(Prompt)

        /// A tool call containing a tool name and the arguments to invoke it with.
        case toolCalls(ToolCalls)

        /// An tool output provided back to the model.
        case toolOutput(ToolOutput)

        /// A response from the model.
        case response(Response)

        /// The stable identity of the entity associated with this instance.
        public var id: String {
            switch self {
            case .instructions(let instructions):
                return instructions.id
            case .prompt(let prompt):
                return prompt.id
            case .toolCalls(let toolCalls):
                return toolCalls.id
            case .toolOutput(let toolOutput):
                return toolOutput.id
            case .response(let response):
                return response.id
            }
        }
    }

    /// The types of segments that may be included in a transcript entry.
    public enum Segment: Sendable, Identifiable, Equatable, Codable {
        /// A segment containing text.
        case text(TextSegment)

        /// A segment containing structured content
        case structure(StructuredSegment)

        /// The stable identity of the entity associated with this instance.
        public var id: String {
            switch self {
            case .text(let textSegment):
                return textSegment.id
            case .structure(let structuredSegment):
                return structuredSegment.id
            }
        }
    }

    /// A segment containing text.
    public struct TextSegment: Sendable, Identifiable, Equatable, Codable {
        /// The stable identity of the entity associated with this instance.
        public var id: String

        public var content: String

        public init(id: String = UUID().uuidString, content: String) {
            self.id = id
            self.content = content
        }
    }

    /// A segment containing structured content.
    public struct StructuredSegment: Sendable, Identifiable, Equatable, Codable {
        /// The stable identity of the entity associated with this instance.
        public var id: String

        /// A source that be used to understand which type content represents.
        public var source: String

        /// The content of the segment.
        public var content: GeneratedContent

        public init(id: String = UUID().uuidString, source: String, content: GeneratedContent) {
            self.id = id
            self.source = source
            self.content = content
        }
    }

    /// Instructions you provide to the model that define its behavior.
    ///
    /// Instructions are typically provided to define the role and behavior of the model. Apple trains the model
    /// to obey instructions over any commands it receives in prompts. This is a security mechanism to help
    /// mitigate prompt injection attacks.
    public struct Instructions: Sendable, Identifiable, Equatable, Codable {
        /// The stable identity of the entity associated with this instance.
        public var id: String

        /// The content of the instructions, in natural language.
        ///
        /// - Note: Instructions are often provided in English even when the
        /// users interact with the model in another language.
        public var segments: [Segment]

        /// A list of tools made available to the model.
        public var toolDefinitions: [ToolDefinition]

        /// Initialize instructions by describing how you want the model to
        /// behave using natural language.
        ///
        /// - Parameters:
        ///   - id: A unique identifier for this instructions segment.
        ///   - segments: An array of segments that make up the instructions.
        ///   - toolDefinitions: Tools that the model should be allowed to call.
        public init(
            id: String = UUID().uuidString,
            segments: [Segment],
            toolDefinitions: [ToolDefinition]
        ) {
            self.id = id
            self.segments = segments
            self.toolDefinitions = toolDefinitions
        }
    }

    /// A prompt from the user asking the model.
    public struct Prompt: Sendable, Identifiable, Equatable, Codable {
        /// The identifier of the prompt.
        public var id: String

        /// Ordered prompt segments.
        public var segments: [Segment]

        /// Generation options associated with the prompt.
        public var options: GenerationOptions

        /// An optional response format that describes the desired output structure.
        public var responseFormat: ResponseFormat?

        /// Creates a prompt.
        ///
        /// - Parameters:
        ///   - id: A ``Generable`` type to use as the response format.
        ///   - segments: An array of segments that make up the prompt.
        ///   - options: Options that control how tokens are sampled from the distribution the model produces.
        ///   - responseFormat: A response format that describes the output structure.
        public init(
            id: String = UUID().uuidString,
            segments: [Segment],
            options: GenerationOptions = GenerationOptions(),
            responseFormat: ResponseFormat? = nil
        ) {
            self.id = id
            self.segments = segments
            self.options = options
            self.responseFormat = responseFormat
        }
    }

    /// Specifies a response format that the model must conform its output to.
    public struct ResponseFormat: Sendable, Codable {
        private let schema: GenerationSchema

        /// A name associated with the response format.
        public var name: String {
            // Extract type name from the schema's debug description
            // This is a best-effort approach
            let desc = schema.debugDescription
            if let range = desc.range(of: "$ref("),
                let endRange = desc.range(of: ")", range: range.upperBound ..< desc.endIndex)
            {
                let name = desc[range.upperBound ..< endRange.lowerBound]
                return String(name)
            }
            return "response"
        }

        /// Creates a response format with type you specify.
        ///
        /// - Parameters:
        ///   - type: A ``Generable`` type to use as the response format.
        public init<Content>(type: Content.Type) where Content: Generable {
            self.schema = Content.generationSchema
        }

        /// Creates a response format with a schema.
        ///
        /// - Parameters:
        ///   - schema: A schema to use as the response format.
        public init(schema: GenerationSchema) {
            self.schema = schema
        }
    }

    /// A collection tool calls generated by the model.
    public struct ToolCalls: Sendable, Identifiable, Equatable, Codable {
        /// The stable identity of the entity associated with this instance.
        public var id: String

        private var calls: [ToolCall]

        public init<S>(id: String = UUID().uuidString, _ calls: S)
        where S: Sequence, S.Element == ToolCall {
            self.id = id
            self.calls = Array(calls)
        }
    }

    /// A tool call generated by the model containing the name of a tool and arguments to pass to it.
    public struct ToolCall: Sendable, Identifiable, Equatable, Codable {
        /// The stable identity of the entity associated with this instance.
        public var id: String

        /// The name of the tool being invoked.
        public var toolName: String

        /// Arguments to pass to the invoked tool.
        public var arguments: GeneratedContent

        public init(id: String, toolName: String, arguments: GeneratedContent) {
            self.id = id
            self.toolName = toolName
            self.arguments = arguments
        }
    }

    /// A tool output provided back to the model.
    public struct ToolOutput: Sendable, Identifiable, Equatable, Codable {
        /// A unique id for this tool output.
        public var id: String

        /// The name of the tool that produced this output.
        public var toolName: String

        /// Segments of the tool output.
        public var segments: [Segment]

        public init(id: String, toolName: String, segments: [Segment]) {
            self.id = id
            self.toolName = toolName
            self.segments = segments
        }
    }

    /// A response from the model.
    public struct Response: Sendable, Identifiable, Equatable, Codable {
        /// The stable identity of the entity associated with this instance.
        public var id: String

        /// Version aware identifiers for all assets used to generate this response.
        public var assetIDs: [String]

        /// Ordered prompt segments.
        public var segments: [Segment]

        public init(
            id: String = UUID().uuidString,
            assetIDs: [String],
            segments: [Segment]
        ) {
            self.id = id
            self.assetIDs = assetIDs
            self.segments = segments
        }
    }

    /// A definition of a tool.
    public struct ToolDefinition: Sendable, Codable {
        /// The tool's name.
        public var name: String

        /// A description of how and when to use the tool.
        public var description: String

        private let parameters: GenerationSchema

        public init(name: String, description: String, parameters: GenerationSchema) {
            self.name = name
            self.description = description
            self.parameters = parameters
        }

        public init(tool: some Tool) {
            self.name = tool.name
            self.description = tool.description
            self.parameters = tool.parameters
        }
    }
}

// MARK: - CustomStringConvertible

extension Transcript.Entry: CustomStringConvertible {
    public var description: String {
        switch self {
        case .instructions(let instructions):
            return "instructions(\(instructions))"
        case .prompt(let prompt):
            return "prompt(\(prompt))"
        case .toolCalls(let toolCalls):
            return "toolCalls(\(toolCalls))"
        case .toolOutput(let toolOutput):
            return "toolOutput(\(toolOutput))"
        case .response(let response):
            return "response(\(response))"
        }
    }
}

extension Transcript.Segment: CustomStringConvertible {
    public var description: String {
        switch self {
        case .text(let textSegment):
            return textSegment.description
        case .structure(let structuredSegment):
            return structuredSegment.description
        }
    }
}

extension Transcript.TextSegment: CustomStringConvertible {
    public var description: String { content }
}

extension Transcript.StructuredSegment: CustomStringConvertible {
    public var description: String {
        "StructuredSegment(source: \(source), content: \(content))"
    }
}

extension Transcript.Instructions: CustomStringConvertible {
    public var description: String {
        "Instructions(segments: \(segments.count), tools: \(toolDefinitions.count))"
    }
}

extension Transcript.Prompt: CustomStringConvertible {
    public var description: String {
        "Prompt(segments: \(segments.count))"
    }
}

extension Transcript.ResponseFormat: CustomStringConvertible {
    public var description: String {
        "ResponseFormat(name: \(name))"
    }
}

extension Transcript.ToolCalls: CustomStringConvertible {
    public var description: String {
        "ToolCalls(\(count) calls)"
    }
}

extension Transcript.ToolCall: CustomStringConvertible {
    public var description: String {
        "ToolCall(tool: \(toolName))"
    }
}

extension Transcript.ToolOutput: CustomStringConvertible {
    public var description: String {
        "ToolOutput(tool: \(toolName), segments: \(segments.count))"
    }
}

extension Transcript.Response: CustomStringConvertible {
    public var description: String {
        "Response(segments: \(segments.count))"
    }
}

// MARK: - Equatable

extension Transcript.ResponseFormat: Equatable {
    public static func == (lhs: Transcript.ResponseFormat, rhs: Transcript.ResponseFormat) -> Bool {
        return lhs.name == rhs.name
    }
}

extension Transcript.ToolDefinition: Equatable {
    public static func == (lhs: Transcript.ToolDefinition, rhs: Transcript.ToolDefinition) -> Bool {
        return lhs.name == rhs.name && lhs.description == rhs.description
    }
}

// MARK: - RandomAccessCollection

extension Transcript: RandomAccessCollection {
    public subscript(index: Int) -> Entry {
        entries[index]
    }

    public var startIndex: Int {
        entries.startIndex
    }

    public var endIndex: Int {
        entries.endIndex
    }
}

extension Transcript.ToolCalls: RandomAccessCollection {
    public subscript(position: Int) -> Transcript.ToolCall {
        calls[position]
    }

    public var startIndex: Int {
        calls.startIndex
    }

    public var endIndex: Int {
        calls.endIndex
    }
}
