import EventSource
import Foundation
import JSONSchema
import OrderedCollections

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

/// A language model that connects to Anthropic's Claude API.
///
/// Use this model to generate text using Claude models from Anthropic.
///
/// ```swift
/// let model = AnthropicLanguageModel(
///     apiKey: "your-api-key",
///     model: "claude-3-5-sonnet-20241022"
/// )
/// ```
///
/// You can also specify beta headers to access experimental features:
///
/// ```swift
/// let model = AnthropicLanguageModel(
///     apiKey: "your-api-key",
///     model: "claude-3-5-sonnet-20241022",
///     betas: ["beta1", "beta2"]
/// )
/// ```
public struct AnthropicLanguageModel: LanguageModel {
    /// The reason the model is unavailable.
    /// This model is always available.
    public typealias UnavailableReason = Never

    /// The default base URL for Anthropic's API.
    public static let defaultBaseURL = URL(string: "https://api.anthropic.com")!

    /// The default API version for Anthropic's API.
    public static let defaultAPIVersion = "2023-06-01"

    /// The base URL for the API endpoint.
    public let baseURL: URL

    /// The closure providing the API key for authentication.
    private let tokenProvider: @Sendable () -> String

    /// The API version to use for requests.
    public let apiVersion: String

    /// Optional beta version(s) of the API to use.
    public let betas: [String]?

    /// The model identifier to use for generation.
    public let model: String

    private let urlSession: URLSession

    /// Creates an Anthropic language model.
    ///
    /// - Parameters:
    ///   - baseURL: The base URL for the API endpoint. Defaults to Anthropic's official API.
    ///   - apiKey: Your Anthropic API key or a closure that returns it.
    ///   - apiVersion: The API version to use for requests. Defaults to `2023-06-01`.
    ///   - betas: Optional beta version(s) of the API to use.
    ///   - model: The model identifier (for example, "claude-3-5-sonnet-20241022").
    ///   - session: The URL session to use for network requests.
    public init(
        baseURL: URL = defaultBaseURL,
        apiKey tokenProvider: @escaping @autoclosure @Sendable () -> String,
        apiVersion: String = defaultAPIVersion,
        betas: [String]? = nil,
        model: String,
        session: URLSession = URLSession(configuration: .default)
    ) {
        var baseURL = baseURL
        if !baseURL.path.hasSuffix("/") {
            baseURL = baseURL.appendingPathComponent("")
        }

        self.baseURL = baseURL
        self.tokenProvider = tokenProvider
        self.apiVersion = apiVersion
        self.betas = betas
        self.model = model
        self.urlSession = session
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
            fatalError("AnthropicLanguageModel only supports generating String content")
        }

        let url = baseURL.appendingPathComponent("v1/messages")
        let headers = buildHeaders()

        let messages = [
            AnthropicMessage(role: .user, content: [.text(.init(text: prompt.description))])
        ]

        // Convert available tools to Anthropic format
        let anthropicTools: [AnthropicTool] = try session.tools.map { tool in
            try convertToolToAnthropicFormat(tool)
        }

        let params = try createMessageParams(
            model: model,
            system: nil,
            messages: messages,
            tools: anthropicTools.isEmpty ? nil : anthropicTools,
            options: options
        )

        let body = try JSONEncoder().encode(params)

        let message: AnthropicMessageResponse = try await urlSession.fetch(
            .post,
            url: url,
            headers: headers,
            body: body
        )

        var entries: [Transcript.Entry] = []

        // Handle tool calls, if present
        let toolUses: [AnthropicToolUse] = message.content.compactMap { block in
            if case .toolUse(let u) = block { return u }
            return nil
        }

        if !toolUses.isEmpty {
            let invocations = try await resolveToolUses(toolUses, session: session)
            if !invocations.isEmpty {
                entries.append(.toolCalls(Transcript.ToolCalls(invocations.map(\.call))))
                for invocation in invocations {
                    entries.append(.toolOutput(invocation.output))
                }
            }
        }

        let text = message.content.compactMap { block -> String? in
            switch block {
            case .text(let t): return t.text
            default: return nil
            }
        }.joined()

        return LanguageModelSession.Response(
            content: text as! Content,
            rawContent: GeneratedContent(text),
            transcriptEntries: ArraySlice(entries)
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
            fatalError("AnthropicLanguageModel only supports generating String content")
        }

        let messages = [
            AnthropicMessage(role: .user, content: [.text(.init(text: prompt.description))])
        ]

        let url = baseURL.appendingPathComponent("v1/messages")

        let stream: AsyncThrowingStream<LanguageModelSession.ResponseStream<Content>.Snapshot, any Error> = .init {
            continuation in
            let task = Task { @Sendable in
                do {
                    let headers = buildHeaders()

                    // Convert available tools to Anthropic format
                    let anthropicTools: [AnthropicTool] = try session.tools.map { tool in
                        try convertToolToAnthropicFormat(tool)
                    }

                    var params = try createMessageParams(
                        model: model,
                        system: nil,
                        messages: messages,
                        tools: anthropicTools.isEmpty ? nil : anthropicTools,
                        options: options
                    )
                    params["stream"] = .bool(true)

                    let body = try JSONEncoder().encode(params)

                    // Stream server-sent events from Anthropic API
                    let events: AsyncThrowingStream<AnthropicStreamEvent, any Error> =
                        urlSession
                        .fetchEventStream(
                            .post,
                            url: url,
                            headers: headers,
                            body: body
                        )

                    var accumulatedText = ""

                    for try await event in events {
                        switch event {
                        case .contentBlockDelta(let delta):
                            if case .textDelta(let textDelta) = delta.delta {
                                accumulatedText += textDelta.text

                                // Yield snapshot with partially generated content
                                let raw = GeneratedContent(accumulatedText)
                                let content: Content.PartiallyGenerated = (accumulatedText as! Content)
                                    .asPartiallyGenerated()
                                continuation.yield(.init(content: content, rawContent: raw))
                            }
                        case .messageStop:
                            continuation.finish()
                            return
                        case .messageStart, .contentBlockStart, .contentBlockStop, .messageDelta, .ping, .ignored:
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

    private func buildHeaders() -> [String: String] {
        var headers: [String: String] = [
            "x-api-key": tokenProvider(),
            "anthropic-version": apiVersion,
        ]

        if let betas = betas, !betas.isEmpty {
            headers["anthropic-beta"] = betas.joined(separator: ",")
        }

        return headers
    }
}

// MARK: - Conversions

private func createMessageParams(
    model: String,
    system: String?,
    messages: [AnthropicMessage],
    tools: [AnthropicTool]?,
    options: GenerationOptions
) throws -> [String: JSONValue] {
    var params: [String: JSONValue] = [
        "model": .string(model),
        "messages": try JSONValue(messages),
        "max_tokens": .int(options.maximumResponseTokens ?? 1024),
    ]

    if let system {
        params["system"] = .string(system)
    }
    if let tools, !tools.isEmpty {
        params["tools"] = try JSONValue(tools)
    }
    if let temperature = options.temperature {
        params["temperature"] = .double(temperature)
    }

    return params
}

// MARK: - Tool Invocation Handling

private struct ToolInvocationResult {
    let call: Transcript.ToolCall
    let output: Transcript.ToolOutput
}

private func resolveToolUses(
    _ toolUses: [AnthropicToolUse],
    session: LanguageModelSession
) async throws -> [ToolInvocationResult] {
    if toolUses.isEmpty { return [] }

    var toolsByName: [String: any Tool] = [:]
    for tool in session.tools {
        if toolsByName[tool.name] == nil {
            toolsByName[tool.name] = tool
        }
    }

    var results: [ToolInvocationResult] = []
    results.reserveCapacity(toolUses.count)

    for use in toolUses {
        let args = try toGeneratedContent(use.input)
        let callID = use.id
        let transcriptCall = Transcript.ToolCall(
            id: callID,
            toolName: use.name,
            arguments: args
        )

        guard let tool = toolsByName[use.name] else {
            let message = Transcript.Segment.text(.init(content: "Tool not found: \(use.name)"))
            let output = Transcript.ToolOutput(
                id: callID,
                toolName: use.name,
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

// Convert our GenerationSchema into Anthropic's expected JSON Schema payload
private func convertToolToAnthropicFormat(_ tool: any Tool) throws -> AnthropicTool {
    // Resolve the schema root to ensure it has a type field (Anthropic requirement)
    let resolvedSchema = tool.parameters.withResolvedRoot() ?? tool.parameters

    // Encode our internal schema then decode to JSONSchema type
    let data = try JSONEncoder().encode(resolvedSchema)
    let schema = try JSONDecoder().decode(JSONSchema.self, from: data)
    return AnthropicTool(name: tool.name, description: tool.description, inputSchema: schema)
}

private func toGeneratedContent(_ value: [String: JSONValue]?) throws -> GeneratedContent {
    guard let value else { return GeneratedContent(properties: [:]) }
    let data = try JSONEncoder().encode(JSONValue.object(value))
    let json = String(data: data, encoding: .utf8) ?? "{}"
    return try GeneratedContent(json: json)
}

// MARK: - Supporting Types

private struct AnthropicTool: Codable, Sendable {
    let name: String
    let description: String
    let inputSchema: JSONSchema

    enum CodingKeys: String, CodingKey {
        case name
        case description
        case inputSchema = "input_schema"
    }
}

private struct AnthropicMessage: Codable, Sendable {
    enum Role: String, Codable, Sendable { case user, assistant }

    let role: Role
    let content: [AnthropicContent]
}

private enum AnthropicContent: Codable, Sendable {
    case text(AnthropicText)
    case toolUse(AnthropicToolUse)

    enum CodingKeys: String, CodingKey { case type }

    enum ContentType: String, Codable { case text = "text", toolUse = "tool_use" }

    init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(ContentType.self, forKey: .type)
        switch type {
        case .text:
            self = .text(try AnthropicText(from: decoder))
        case .toolUse:
            self = .toolUse(try AnthropicToolUse(from: decoder))
        }
    }

    func encode(to encoder: any Encoder) throws {
        switch self {
        case .text(let t): try t.encode(to: encoder)
        case .toolUse(let u): try u.encode(to: encoder)
        }
    }
}

private struct AnthropicText: Codable, Sendable {
    let type: String
    let text: String

    init(text: String) {
        self.type = "text"
        self.text = text
    }
}

private struct AnthropicToolUse: Codable, Sendable {
    let type: String
    let id: String
    let name: String
    let input: [String: JSONValue]?

    init(id: String, name: String, input: [String: JSONValue]?) {
        self.type = "tool_use"
        self.id = id
        self.name = name
        self.input = input
    }
}

private struct AnthropicMessageResponse: Codable, Sendable {
    let id: String
    let type: String
    let role: String
    let content: [AnthropicContent]
    let model: String
    let stopReason: StopReason?

    enum CodingKeys: String, CodingKey {
        case id, type, role, content, model
        case stopReason = "stop_reason"
    }

    enum StopReason: String, Codable {
        case endTurn = "end_turn"
        case maxTokens = "max_tokens"
        case stopSequence = "stop_sequence"
        case toolUse = "tool_use"
        case pauseTurn = "pause_turn"
        case refusal = "refusal"
        case modelContextWindowExceeded = "model_context_window_exceeded"
    }
}

private struct AnthropicErrorResponse: Codable { let error: AnthropicErrorDetail }
private struct AnthropicErrorDetail: Codable {
    let type: String
    let message: String
}

// MARK: - Streaming Event Types

private enum AnthropicStreamEvent: Codable, Sendable {
    case messageStart(MessageStartEvent)
    case contentBlockStart(ContentBlockStartEvent)
    case contentBlockDelta(ContentBlockDeltaEvent)
    case contentBlockStop(ContentBlockStopEvent)
    case messageDelta(MessageDeltaEvent)
    case messageStop
    case ping
    case ignored

    enum CodingKeys: String, CodingKey { case type }

    init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)

        switch type {
        case "message_start":
            self = .messageStart(try MessageStartEvent(from: decoder))
        case "content_block_start":
            self = .contentBlockStart(try ContentBlockStartEvent(from: decoder))
        case "content_block_delta":
            self = .contentBlockDelta(try ContentBlockDeltaEvent(from: decoder))
        case "content_block_stop":
            self = .contentBlockStop(try ContentBlockStopEvent(from: decoder))
        case "message_delta":
            self = .messageDelta(try MessageDeltaEvent(from: decoder))
        case "message_stop":
            self = .messageStop
        case "ping":
            self = .ping
        default:
            self = .ignored
        }
    }

    func encode(to encoder: any Encoder) throws {
        switch self {
        case .messageStart(let event): try event.encode(to: encoder)
        case .contentBlockStart(let event): try event.encode(to: encoder)
        case .contentBlockDelta(let event): try event.encode(to: encoder)
        case .contentBlockStop(let event): try event.encode(to: encoder)
        case .messageDelta(let event): try event.encode(to: encoder)
        case .messageStop:
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode("message_stop", forKey: .type)
        case .ping:
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode("ping", forKey: .type)
        case .ignored:
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode("ignored", forKey: .type)
        }
    }

    struct MessageStartEvent: Codable, Sendable {
        let type: String
        let message: AnthropicMessageResponse
    }

    struct ContentBlockStartEvent: Codable, Sendable {
        let type: String
        let index: Int
        let contentBlock: ContentBlock

        enum CodingKeys: String, CodingKey {
            case type, index
            case contentBlock = "content_block"
        }

        struct ContentBlock: Codable, Sendable {
            let type: String
            let text: String?
        }
    }

    struct ContentBlockDeltaEvent: Codable, Sendable {
        let type: String
        let index: Int
        let delta: Delta

        enum Delta: Codable, Sendable {
            case textDelta(TextDelta)
            case inputJsonDelta(InputJsonDelta)
            case ignored

            enum CodingKeys: String, CodingKey { case type }

            init(from decoder: any Decoder) throws {
                let container = try decoder.container(keyedBy: CodingKeys.self)
                let type = try container.decode(String.self, forKey: .type)

                switch type {
                case "text_delta":
                    self = .textDelta(try TextDelta(from: decoder))
                case "input_json_delta":
                    self = .inputJsonDelta(try InputJsonDelta(from: decoder))
                default:
                    self = .ignored
                }
            }

            func encode(to encoder: any Encoder) throws {
                switch self {
                case .textDelta(let delta): try delta.encode(to: encoder)
                case .inputJsonDelta(let delta): try delta.encode(to: encoder)
                case .ignored:
                    var container = encoder.container(keyedBy: CodingKeys.self)
                    try container.encode("ignored", forKey: .type)
                }
            }

            struct TextDelta: Codable, Sendable {
                let type: String
                let text: String
            }

            struct InputJsonDelta: Codable, Sendable {
                let type: String
                let partialJson: String

                enum CodingKeys: String, CodingKey {
                    case type
                    case partialJson = "partial_json"
                }
            }
        }
    }

    struct ContentBlockStopEvent: Codable, Sendable {
        let type: String
        let index: Int
    }

    struct MessageDeltaEvent: Codable, Sendable {
        let type: String
        let delta: Delta

        struct Delta: Codable, Sendable {
            let stopReason: String?
            let stopSequence: String?

            enum CodingKeys: String, CodingKey {
                case stopReason = "stop_reason"
                case stopSequence = "stop_sequence"
            }
        }
    }
}
