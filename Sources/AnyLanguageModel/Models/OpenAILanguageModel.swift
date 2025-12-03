import Foundation
import JSONSchema

#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// A language model that connects to OpenAI-compatible APIs.
///
/// Use this model to generate text using OpenAI's Chat Completions or Responses APIs.
/// You can specify a custom base URL to work with OpenAI-compatible services.
///
/// ```swift
/// let model = OpenAILanguageModel(
///     apiKey: "your-api-key",
///     model: "gpt-4"
/// )
/// ```
public struct OpenAILanguageModel: LanguageModel {
    /// The reason the model is unavailable.
    /// This model is always available.
    public typealias UnavailableReason = Never

    /// The default base URL for OpenAI's API.
    public static let defaultBaseURL = URL(string: "https://api.openai.com/v1/")!

    /// The OpenAI API variant to use.
    public enum APIVariant: Sendable {
        /// When selected, use the Chat Completions API.
        /// https://platform.openai.com/docs/api-reference/chat/create
        case chatCompletions
        /// When selected, use the Responses API.
        /// https://platform.openai.com/docs/api-reference/responses
        case responses
    }

    /// The base URL for the API endpoint.
    public let baseURL: URL

    /// The closure providing the API key for authentication.
    private let tokenProvider: @Sendable () -> String

    /// The model identifier to use for generation.
    public let model: String

    /// The API variant to use.
    public let apiVariant: APIVariant

    private let urlSession: URLSession

    /// Creates an OpenAI language model.
    ///
    /// - Parameters:
    ///   - baseURL: The base URL for the API endpoint. Defaults to OpenAI's official API.
    ///   - apiKey: Your OpenAI API key or a closure that returns it.
    ///   - model: The model identifier (for example, "gpt-4" or "gpt-3.5-turbo").
    ///   - apiVariant: The API variant to use. Defaults to `.chatCompletions`.
    ///   - session: The URL session to use for network requests.
    public init(
        baseURL: URL = defaultBaseURL,
        apiKey tokenProvider: @escaping @autoclosure @Sendable () -> String,
        model: String,
        apiVariant: APIVariant = .chatCompletions,
        session: URLSession = URLSession(configuration: .default)
    ) {
        var baseURL = baseURL
        if !baseURL.path.hasSuffix("/") {
            baseURL = baseURL.appendingPathComponent("")
        }

        self.baseURL = baseURL
        self.tokenProvider = tokenProvider
        self.model = model
        self.apiVariant = apiVariant
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
            fatalError("OpenAILanguageModel only supports generating String content")
        }

        var messages: [OpenAIMessage] = []
        if let systemSegments = extractInstructionSegments(from: session) {
            messages.append(
                OpenAIMessage(role: .system, content: .blocks(convertSegmentsToOpenAIBlocks(systemSegments)))
            )
        }
        let userSegments = extractPromptSegments(from: session, fallbackText: prompt.description)
        messages.append(OpenAIMessage(role: .user, content: .blocks(convertSegmentsToOpenAIBlocks(userSegments))))

        // Convert tools if any are available in the session
        let openAITools: [OpenAITool]? = {
            guard !session.tools.isEmpty else { return nil }
            var converted: [OpenAITool] = []
            converted.reserveCapacity(session.tools.count)
            for tool in session.tools {
                converted.append(convertToolToOpenAIFormat(tool))
            }
            return converted
        }()

        switch apiVariant {
        case .chatCompletions:
            return try await respondWithChatCompletions(
                messages: messages,
                tools: openAITools,
                options: options,
                session: session
            )
        case .responses:
            return try await respondWithResponses(
                messages: messages,
                tools: openAITools,
                options: options,
                session: session
            )
        }
    }

    private func respondWithChatCompletions<Content>(
        messages: [OpenAIMessage],
        tools: [OpenAITool]?,
        options: GenerationOptions,
        session: LanguageModelSession
    ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {

        var entries: [Transcript.Entry] = []
        var text = ""
        var messages = messages

        // Loop until no more tool calls
        while true {
            let params = ChatCompletions.createRequestBody(
                model: model,
                messages: messages,
                tools: tools,
                options: options,
                stream: false
            )

            let url = baseURL.appendingPathComponent("chat/completions")
            let body = try JSONEncoder().encode(params)
            let resp: ChatCompletions.Response = try await urlSession.fetch(
                .post,
                url: url,
                headers: [
                    "Authorization": "Bearer \(tokenProvider())"
                ],
                body: body
            )

            guard let choice = resp.choices.first else {
                return LanguageModelSession.Response(
                    content: "" as! Content,
                    rawContent: GeneratedContent(""),
                    transcriptEntries: ArraySlice(entries)
                )
            }

            let toolCallMessage = choice.message
            if let toolCalls = toolCallMessage.toolCalls, !toolCalls.isEmpty {
                if let value = try? JSONValue(toolCallMessage) {
                    messages.append(OpenAIMessage(role: .raw(rawContent: value), content: .text("")))
                }
                let invocations = try await resolveToolCalls(toolCalls, session: session)
                if !invocations.isEmpty {
                    entries.append(.toolCalls(Transcript.ToolCalls(invocations.map { $0.call })))
                    for invocation in invocations {
                        let output =  invocation.output
                        entries.append(.toolOutput(output))
                        let toolSegments: [Transcript.Segment] = output.segments
                        let blocks = convertSegmentsToOpenAIBlocks(toolSegments)
                        messages.append(OpenAIMessage(role: .tool(id: invocation.call.id), content: .blocks(blocks)))
                    }
                    continue
                }
            }

            text = choice.message.content ?? ""
            break
        }
        return LanguageModelSession.Response(
            content: text as! Content,
            rawContent: GeneratedContent(text),
            transcriptEntries: ArraySlice(entries)
        )
    }

    private func respondWithResponses<Content>(
        messages: [OpenAIMessage],
        tools: [OpenAITool]?,
        options: GenerationOptions,
        session: LanguageModelSession
    ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
        var entries: [Transcript.Entry] = []
        var text = ""
        var messages = messages

        let url = baseURL.appendingPathComponent("responses")

        // Loop until no more tool calls
        while true {
            let params = Responses.createRequestBody(
                model: model,
                messages: messages,
                tools: tools,
                options: options,
                stream: false
            )

            var encoder = JSONEncoder()
            let body = try encoder.encode(params)
            let resp: Responses.Response = try await urlSession.fetch(
                .post,
                url: url,
                headers: [
                    "Authorization": "Bearer \(tokenProvider())"
                ],
                body: body
            )

            let toolCalls = extractToolCallsFromOutput(resp.output)
            if !toolCalls.isEmpty {
                if let output = resp.output {
                    for msg in output {
                        messages.append(OpenAIMessage(role: .raw(rawContent: msg), content: .text("")))
                    }
                }
                let invocations = try await resolveToolCalls(toolCalls, session: session)
                if !invocations.isEmpty {
                    entries.append(.toolCalls(Transcript.ToolCalls(invocations.map { $0.call })))

                    for invocation in invocations {
                        let output =  invocation.output
                        entries.append(.toolOutput(output))
                        let toolSegments: [Transcript.Segment] = output.segments
                        let blocks = convertSegmentsToOpenAIBlocks(toolSegments)
                        if blocks.count != 1 {
                            print("blocks count is not 1: \(blocks)")
                        }
                        messages.append(OpenAIMessage(role: .tool(id: invocation.call.id), content: .blocks(blocks)))
                    }
                    continue
                }
            }

            text = resp.outputText ?? extractTextFromOutput(resp.output) ?? ""

            break
        }
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
            fatalError("OpenAILanguageModel only supports generating String content")
        }

        var messages: [OpenAIMessage] = []
        if let systemSegments = extractInstructionSegments(from: session) {
            messages.append(
                OpenAIMessage(role: .system, content: .blocks(convertSegmentsToOpenAIBlocks(systemSegments)))
            )
        }
        let userSegments = extractPromptSegments(from: session, fallbackText: prompt.description)
        messages.append(OpenAIMessage(role: .user, content: .blocks(convertSegmentsToOpenAIBlocks(userSegments))))

        // Convert tools if any are available in the session
        let openAITools: [OpenAITool]? = {
            guard !session.tools.isEmpty else { return nil }
            var converted: [OpenAITool] = []
            converted.reserveCapacity(session.tools.count)
            for tool in session.tools {
                converted.append(convertToolToOpenAIFormat(tool))
            }
            return converted
        }()

        switch apiVariant {
        case .responses:
            let params = Responses.createRequestBody(
                model: model,
                messages: messages,
                tools: openAITools,
                options: options,
                stream: true
            )

            let url = baseURL.appendingPathComponent("responses")

            let stream: AsyncThrowingStream<LanguageModelSession.ResponseStream<Content>.Snapshot, any Error> = .init {
                continuation in
                let task = Task { @Sendable in
                    do {
                        let body = try JSONEncoder().encode(params)

                        let events: AsyncThrowingStream<OpenAIResponsesServerEvent, any Error> =
                        urlSession.fetchEventStream(
                            .post,
                            url: url,
                            headers: [
                                "Authorization": "Bearer \(tokenProvider())"
                            ],
                            body: body
                        )

                        var accumulatedText = ""

                        for try await event in events {
                            switch event {
                            case .outputTextDelta(let delta):
                                accumulatedText += delta

                                // Yield snapshot with partially generated content
                                let raw = GeneratedContent(accumulatedText)
                                let content: Content.PartiallyGenerated = (accumulatedText as! Content)
                                    .asPartiallyGenerated()
                                continuation.yield(.init(content: content, rawContent: raw))

                            case .toolCallCreated(_):
                                // Minimal streaming implementation ignores tool call events
                                break
                            case .toolCallDelta(_):
                                // Minimal streaming implementation ignores tool call deltas
                                break
                            case .completed(_):
                                continuation.finish()
                            case .ignored:
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

        case .chatCompletions:
            let params = ChatCompletions.createRequestBody(
                model: model,
                messages: messages,
                tools: openAITools,
                options: options,
                stream: true
            )

            let url = baseURL.appendingPathComponent("chat/completions")

            let stream: AsyncThrowingStream<LanguageModelSession.ResponseStream<Content>.Snapshot, any Error> = .init {
                continuation in
                let task = Task { @Sendable in
                    do {
                        let body = try JSONEncoder().encode(params)

                        let events: AsyncThrowingStream<OpenAIChatCompletionsChunk, any Error> =
                        urlSession.fetchEventStream(
                            .post,
                            url: url,
                            headers: [
                                "Authorization": "Bearer \(tokenProvider())"
                            ],
                            body: body
                        )

                        var accumulatedText = ""

                        for try await chunk in events {
                            if let choice = chunk.choices.first {
                                if let piece = choice.delta.content, !piece.isEmpty {
                                    accumulatedText += piece

                                    let raw = GeneratedContent(accumulatedText)
                                    let content: Content.PartiallyGenerated = (accumulatedText as! Content)
                                        .asPartiallyGenerated()
                                    continuation.yield(.init(content: content, rawContent: raw))
                                }

                                if choice.finishReason != nil {
                                    continuation.finish()
                                }
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
    }
}

// MARK: - API Variants

private enum ChatCompletions {
    static func createRequestBody(
        model: String,
        messages: [OpenAIMessage],
        tools: [OpenAITool]?,
        options: GenerationOptions,
        stream: Bool
    ) -> JSONValue {
        var body: [String: JSONValue] = [
            "model": .string(model),
            "messages": .array(messages.map { $0.jsonValue(for: .chatCompletions) }),
            "stream": .bool(stream),
        ]

        if let tools {
            body["tools"] = .array(tools.map { $0.jsonValue(for: .chatCompletions) })
        }

        if let temperature = options.temperature {
            body["temperature"] = .double(temperature)
        }
        if let maxTokens = options.maximumResponseTokens {
            body["max_tokens"] = .int(maxTokens)
        }

        return .object(body)
    }

    struct Response: Decodable, Sendable {
        let id: String
        let choices: [Choice]

        struct Choice: Codable, Sendable {
            let message: Message
            let finishReason: String?

            private enum CodingKeys: String, CodingKey {
                case message
                case finishReason = "finish_reason"
            }
        }

        struct Message: Codable, Sendable {
            let role: String
            let content: String?
            let toolCalls: [OpenAIToolCall]?

            private enum CodingKeys: String, CodingKey {
                case role
                case content
                case toolCalls = "tool_calls"
            }
        }
    }
}

private enum Responses {
    static func createRequestBody(
        model: String,
        messages: [OpenAIMessage],
        tools: [OpenAITool]?,
        options: GenerationOptions,
        stream: Bool
    ) -> JSONValue {
        // Build input blocks from the user message content

        var body: [String: JSONValue] = [
            "model": .string(model),
            "stream": .bool(stream),
        ]

        var outputs: [JSONValue] = []
        for msg in messages {
            switch msg.role {
            case .user:

                let userMessage = msg
                // Wrap user content into a single top-level message as required by Responses API
                let contentBlocks: [JSONValue]
                switch userMessage.content {
                case .text(let text):
                    contentBlocks = [
                        .object(["type": .string("input_text"), "text": .string(text)])
                    ]
                case .blocks(let blocks):
                    contentBlocks = blocks.map { block in
                        switch block {
                        case .text(let text):
                            return .object(["type": .string("input_text"), "text": .string(text)])
                        case .imageURL(let url):
                            return .object([
                                "type": .string("input_image"),
                                "image_url": .object(["url": .string(url)]),
                            ])
                        }
                    }
                }
                let object = JSONValue.object([
                    "type": .string("message"),
                    "role": .string("user"),
                    "content": .array(contentBlocks),
                ])
                outputs.append(object)

            case .tool(let id):
                let toolMessage = msg
                // Wrap user content into a single top-level message as required by Responses API
                var contentBlocks: [JSONValue]
                switch toolMessage.content {
                case .text(let text):
                    contentBlocks = [
                        .object(["type": .string("input_text"), "text": .string(text)])
                    ]
                case .blocks(let blocks):
                    contentBlocks = blocks.map { block in
                        switch block {
                        case .text(let text):
                            return .object(["type": .string("input_text"), "text": .string(text)])
                        case .imageURL(let url):
                            return .object([
                                "type": .string("input_image"),
                                "image_url": .object(["url": .string(url)]),
                            ])
                        }
                    }
                }
                if contentBlocks.count > 1 {
                    outputs.append(.object([
                        "type": .string("function_call_output"),
                        "call_id": .string(id),
                        "output": .array(contentBlocks),
                    ]))
                } else {
                    if let object = contentBlocks.first {
                        outputs.append(.object([
                            "type": .string("function_call_output"),
                            "call_id": .string(id),
                            "output":  object,
                       ]))
                    }
                }

            case .raw(rawContent: let rawContent):
                outputs.append(rawContent)


            case .system:
                let systemMessage = msg
                switch systemMessage.content {
                case .text(let text):
                    body["instructions"] = .string(text)
                case .blocks(let blocks):
                    // Concatenate text blocks for instructions; ignore images
                    let text = blocks.compactMap { if case .text(let t) = $0 { return t } else { return nil } }.joined(
                        separator: "\n"
                    )
                    if !text.isEmpty { body["instructions"] = .string(text) }
                }

            case .assistant:
                break
            }
        }
        body["input"] = .array(outputs)

        if let tools {
            body["tools"] = .array(tools.map { $0.jsonValue(for: .responses) })
        }

        if let temperature = options.temperature {
            body["temperature"] = .double(temperature)
        }
        if let maxTokens = options.maximumResponseTokens {
            body["max_output_tokens"] = .int(maxTokens)
        }

        return .object(body)
    }

    struct Response: Decodable, Sendable {
        let id: String
        let output: [JSONValue]?
        let error: [JSONValue]?
        let outputText: String?
        let finishReason: String?

        private enum CodingKeys: String, CodingKey {
            case id
            case output
            case outputText = "output_text"
            case finishReason = "finish_reason"
            case error = "error"
        }
    }
}

// MARK: - Supporting Types

private struct OpenAIMessage: Hashable, Codable, Sendable {
    enum Role: Hashable, Codable, Sendable {
        case system, user, assistant, raw(rawContent: JSONValue), tool(id: String)

        var description: String {
            switch self {
            case .system: return "system"
            case .user: return "user"
            case .assistant: return "assistant"
            case .tool(id: _): return "tool"
            case .raw(rawContent: _): return "raw"
            }
        }
    }

    enum Content: Hashable, Codable, Sendable {
        case text(String)
        case blocks([Block])
    }

    let role: Role
    let content: Content

    func contentAsJsonValue(for apiVariant: OpenAILanguageModel.APIVariant) -> JSONValue {
        switch content {
        case .text(let text):
            switch apiVariant {
            case .chatCompletions:
                return .string(text)
            case .responses:
                return .array([.object(["type": .string("text"), "text": .string(text)])])
            }
        case .blocks(let blocks):
            switch apiVariant {
            case .chatCompletions:
                return .array(blocks.map { $0.jsonValueForChatCompletions })
            case .responses:
                return .array(blocks.map { $0.jsonValueForResponses })
            }
        }
    }

    func jsonValue(for apiVariant: OpenAILanguageModel.APIVariant) -> JSONValue {

        switch role {
        case .raw(rawContent: let rawContent):
            return rawContent

        case .tool(id: let id):
            switch apiVariant {
            case .chatCompletions:
                return .object([
                    "role": .string(role.description),
                    "tool_call_id": .string(id),
                    "content": contentAsJsonValue(for: apiVariant),
                ])
            case .responses:
                return .object([
                    "type": .string("function_call_output"),
                    "call_id": .string(id),
                    "content": contentAsJsonValue(for: apiVariant),
                ])
            }

        case .system, .user, .assistant:
            return .object([
                "role": .string(role.description),
                "content": contentAsJsonValue(for: apiVariant),
            ])
        }
    }


}

private enum Block: Hashable, Codable, Sendable {
    case text(String)
    case imageURL(String)

    var jsonValueForChatCompletions: JSONValue {
        switch self {
        case .text(let text):
            return .object(["type": .string("text"), "text": .string(text)])
        case .imageURL(let url):
            return .object([
                "type": .string("image_url"),
                "image_url": .object(["url": .string(url)]),
            ])
        }
    }

    var jsonValueForResponses: JSONValue {
        switch self {
        case .text(let text):
            return .object(["type": .string("text"), "text": .string(text)])
        case .imageURL(let url):
            // Responses API uses input_image at top-level input, but inside messages we mirror block
            return .object([
                "type": .string("input_image"),
                "image_url": .object(["url": .string(url)]),
            ])
        }
    }
}

private func convertSegmentsToOpenAIBlocks(_ segments: [Transcript.Segment]) -> [Block] {
    var blocks: [Block] = []
    blocks.reserveCapacity(segments.count)
    for segment in segments {
        switch segment {
        case .text(let text):
            blocks.append(.text(text.content))
        case .structure(let structured):
            switch structured.content.kind {
                case .string(let text):
                    blocks.append(.text(text))
            default:
                blocks.append(.text(structured.content.jsonString))
            }
        case .image(let image):
            switch image.source {
            case .url(let url):
                blocks.append(.imageURL(url.absoluteString))
            case .data(let data, let mimeType):
                let b64 = data.base64EncodedString()
                let dataURL = "data:\(mimeType);base64,\(b64)"
                blocks.append(.imageURL(dataURL))
            }
        }
    }
    return blocks
}

private func extractPromptSegments(from session: LanguageModelSession, fallbackText: String) -> [Transcript.Segment] {
    // Prefer the most recent Transcript.Prompt entry if present
    for entry in session.transcript.reversed() {
        if case .prompt(let p) = entry {
            return p.segments
        }
    }
    return [.text(.init(content: fallbackText))]
}

private func extractInstructionSegments(from session: LanguageModelSession) -> [Transcript.Segment]? {
    // Prefer the first Transcript.Instructions entry if present
    for entry in session.transcript {
        if case .instructions(let i) = entry {
            return i.segments
        }
    }
    if let instructions = session.instructions?.description, !instructions.isEmpty {
        return [.text(.init(content: instructions))]
    }
    return nil
}

private struct OpenAITool: Hashable, Codable, Sendable {
    let type: String
    let function: OpenAIFunction

    func jsonValue(for apiVariant: OpenAILanguageModel.APIVariant) -> JSONValue {
        switch apiVariant {
        case .chatCompletions:
            return .object([
                "type": .string(type),
                "function": function.jsonValue,
            ])
        case .responses:
            // Responses API expects name, description, and parameters at the top level
            var obj: [String: JSONValue] = [
                "type": .string(type),
                "name": .string(function.name),
                "description": .string(function.description),
            ]
            if let rawParameters = function.rawParameters {
                obj["parameters"] = rawParameters
            } else if let parameters = function.parameters {
                obj["parameters"] = parameters.jsonValue
            }
            return .object(obj)
        }
    }
}

private struct OpenAIFunction: Hashable, Codable, Sendable {
    let name: String
    let description: String
    let parameters: OpenAIParameters?
    // When available, prefer passing raw JSON Schema converted from GenerationSchema
    // to preserve nested object structures.
    let rawParameters: JSONValue?

    var jsonValue: JSONValue {
        var obj: [String: JSONValue] = [
            "name": .string(name),
            "description": .string(description),
        ]
        if let rawParameters {
            obj["parameters"] = rawParameters
        } else if let parameters {
            obj["parameters"] = parameters.jsonValue
        }
        return .object(obj)
    }
}

private struct OpenAIParameters: Hashable, Codable, Sendable {
    let type: String
    let properties: [String: OpenAISchema]
    let required: [String]

    var jsonValue: JSONValue {
        return .object([
            "type": .string(type),
            "properties": .object(properties.mapValues { $0.jsonValue }),
            "required": .array(required.map { .string($0) }),
        ])
    }
}

private struct OpenAISchema: Hashable, Codable, Sendable {
    let type: String
    let description: String?
    let enumValues: [String]?

    var jsonValue: JSONValue {
        var obj: [String: JSONValue] = ["type": .string(type)]
        if let description { obj["description"] = .string(description) }
        if let enumValues { obj["enum"] = .array(enumValues.map { .string($0) }) }
        return .object(obj)
    }
}

private struct OpenAIToolCall: Codable, Sendable {
    let id: String?
    let type: String?
    let function: OpenAIToolFunction?
}

private struct OpenAIToolFunction: Codable, Sendable {
    let name: String
    let arguments: String?
}

private enum OpenAIResponsesServerEvent: Decodable, Sendable {
    case outputTextDelta(String)
    case toolCallCreated(OpenAIToolCall)
    case toolCallDelta(OpenAIToolCall)
    case completed(String)
    case ignored

    init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decodeIfPresent(String.self, forKey: .type)
        switch type {
        case "response.output_text.delta":
            self = .outputTextDelta(try container.decode(String.self, forKey: .delta))
        case "response.tool_call.created":
            self = .toolCallCreated(try container.decode(OpenAIToolCall.self, forKey: .toolCall))
        case "response.tool_call.delta":
            self = .toolCallDelta(try container.decode(OpenAIToolCall.self, forKey: .toolCall))
        case "response.completed":
            self = .completed((try? container.decode(String.self, forKey: .finishReason)) ?? "stop")
        default:
            self = .ignored
        }
    }

    private enum CodingKeys: String, CodingKey {
        case type
        case delta
        case toolCall = "tool_call"
        case finishReason = "finish_reason"
    }
}

private struct OpenAIChatCompletionsChunk: Decodable, Sendable {
    struct Choice: Decodable, Sendable {
        struct Delta: Decodable, Sendable {
            let role: String?
            let content: String?
        }
        let delta: Delta
        let finishReason: String?

        private enum CodingKeys: String, CodingKey {
            case delta
            case finishReason = "finish_reason"
        }
    }

    let id: String
    let choices: [Choice]
}

private struct OpenAIToolInvocationResult {
    let call: Transcript.ToolCall
    let output: Transcript.ToolOutput
}

private func resolveToolCalls(
    _ toolCalls: [OpenAIToolCall],
    session: LanguageModelSession
) async throws -> [OpenAIToolInvocationResult] {
    if toolCalls.isEmpty { return [] }

    var toolsByName: [String: any Tool] = [:]
    for tool in session.tools {
        if toolsByName[tool.name] == nil {
            toolsByName[tool.name] = tool
        }
    }

    var results: [OpenAIToolInvocationResult] = []
    results.reserveCapacity(toolCalls.count)

    for call in toolCalls {
        guard let function = call.function else { continue }
        let args = try toGeneratedContent(function.arguments)
        let callID = call.id ?? UUID().uuidString
        let transcriptCall = Transcript.ToolCall(
            id: callID,
            toolName: function.name,
            arguments: args
        )

        guard let tool = toolsByName[function.name] else {
            let message = Transcript.Segment.text(.init(content: "Tool not found: \(function.name)"))
            let output = Transcript.ToolOutput(
                id: callID,
                toolName: function.name,
                segments: [message]
            )
            results.append(OpenAIToolInvocationResult(call: transcriptCall, output: output))
            continue
        }

        do {
            let segments = try await tool.makeOutputSegments(from: args)
            let output = Transcript.ToolOutput(
                id: tool.name,
                toolName: tool.name,
                segments: segments
            )
            results.append(OpenAIToolInvocationResult(call: transcriptCall, output: output))
        } catch {
            throw LanguageModelSession.ToolCallError(tool: tool, underlyingError: error)
        }
    }

    return results
}

// MARK: - Converters

private func convertToolToOpenAIFormat(_ tool: any Tool) -> OpenAITool {
    // Prefer passing through a JSONSchema value built from GenerationSchema
    // where possible; fallback to minimal type/required map.
    let rawParameters: JSONValue?

    // Handle the case where the schema has a root reference
    if let resolvedSchema = tool.parameters.withResolvedRoot() {
        rawParameters = try? JSONValue(resolvedSchema)
    } else {
        rawParameters = try? JSONValue(tool.parameters)
    }

    let fn = OpenAIFunction(
        name: tool.name,
        description: tool.description,
        parameters: nil,
        rawParameters: rawParameters
    )
    return OpenAITool(type: "function", function: fn)
}

private func toGeneratedContent(_ jsonString: String?) throws -> GeneratedContent {
    guard let jsonString, !jsonString.isEmpty else { return GeneratedContent(properties: [:]) }
    return try GeneratedContent(json: jsonString)
}

private func extractTextFromOutput(_ output: [JSONValue]?) -> String? {
    guard let output else { return nil }

    var textParts: [String] = []
    for block in output {
        if case let .object(obj) = block,
           case let .string(type)? = obj["type"],
           type == "message",
           case let .array(contentBlocks)? = obj["content"]
        {
            for contentBlock in contentBlocks {
                if case let .object(contentObj) = contentBlock,
                   case let .string(contentType)? = contentObj["type"],
                   contentType == "output_text",
                   case let .string(text)? = contentObj["text"]
                {
                    textParts.append(text)
                }
            }
        }
    }

    return textParts.isEmpty ? nil : textParts.joined()
}

private func extractToolCallsFromOutput(_ output: [JSONValue]?) -> [OpenAIToolCall] {
    guard let output else { return [] }

    var toolCalls: [OpenAIToolCall] = []
    for block in output {
        if case let .object(obj) = block,
           case let .string(type)? = obj["type"]
        {
            // Handle direct function_call at top level
            if type == "function_call" {
                guard let id = obj["id"].flatMap({ if case .string(let s) = $0 { return s } else { return nil } }),
                      let name = obj["name"].flatMap({ if case .string(let s) = $0 { return s } else { return nil } })
                else { continue }

                let argsString: String?
                if let args = obj["arguments"] {
                    if case let .object(argObj) = args {
                        let argsData = try? JSONEncoder().encode(JSONValue.object(argObj))
                        argsString = argsData.flatMap { String(data: $0, encoding: .utf8) }
                    } else if case let .string(str) = args {
                        argsString = str
                    } else {
                        argsString = nil
                    }
                } else {
                    argsString = nil
                }

                let toolCall = OpenAIToolCall(
                    id: id,
                    type: "function",
                    function: OpenAIToolFunction(name: name, arguments: argsString)
                )
                toolCalls.append(toolCall)
            }
            // Handle message with nested content blocks
            else if type == "message", case let .array(contentBlocks)? = obj["content"] {
                for contentBlock in contentBlocks {
                    if case let .object(contentObj) = contentBlock,
                       case let .string(contentType)? = contentObj["type"],
                       (contentType == "tool_call" || contentType == "tool_use")
                    {
                        guard
                            let id = contentObj["id"].flatMap({
                                if case .string(let s) = $0 { return s } else { return nil }
                            }),
                            let name = contentObj["name"].flatMap({
                                if case .string(let s) = $0 { return s } else { return nil }
                            })
                        else { continue }

                        let argsString: String?
                        if let args = contentObj["arguments"] {
                            if case let .object(argObj) = args {
                                let argsData = try? JSONEncoder().encode(JSONValue.object(argObj))
                                argsString = argsData.flatMap { String(data: $0, encoding: .utf8) }
                            } else if case let .string(str) = args {
                                argsString = str
                            } else {
                                argsString = nil
                            }
                        } else if let input = contentObj["input"] {
                            if case let .object(argObj) = input {
                                let argsData = try? JSONEncoder().encode(JSONValue.object(argObj))
                                argsString = argsData.flatMap { String(data: $0, encoding: .utf8) }
                            } else if case let .string(str) = input {
                                argsString = str
                            } else {
                                argsString = nil
                            }
                        } else {
                            argsString = nil
                        }

                        let toolCall = OpenAIToolCall(
                            id: id,
                            type: "function",
                            function: OpenAIToolFunction(name: name, arguments: argsString)
                        )
                        toolCalls.append(toolCall)
                    }
                }
            }
        }
    }

    return toolCalls
}



