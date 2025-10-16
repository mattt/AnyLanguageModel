import Foundation
import JSONSchema

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

public struct OpenAILanguageModel: LanguageModel {
    public static let defaultBaseURL = URL(string: "https://api.openai.com/v1/")!

    public enum APIVariant: Sendable {
        /// When selected, use the Chat Completions API.
        /// https://platform.openai.com/docs/api-reference/chat/create
        case chatCompletions
        /// When selected, use the Responses API.
        /// https://platform.openai.com/docs/api-reference/responses
        case responses
    }

    public let baseURL: URL
    public let apiKey: String
    public let model: String
    public let apiVariant: APIVariant
    private let urlSession: URLSession

    public init(
        baseURL: URL = defaultBaseURL,
        apiKey: String,
        model: String,
        apiVariant: APIVariant = .chatCompletions,
        session: URLSession = URLSession(configuration: .default)
    ) {
        var baseURL = baseURL
        if !baseURL.path.hasSuffix("/") {
            baseURL = baseURL.appendingPathComponent("")
        }

        self.baseURL = baseURL
        self.apiKey = apiKey
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

        let messages = [
            OpenAIMessage(role: .user, content: .text(prompt.description))
        ]

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
                "Authorization": "Bearer \(apiKey)"
            ],
            body: body
        )

        var entries: [Transcript.Entry] = []

        guard let choice = resp.choices.first else {
            return LanguageModelSession.Response(
                content: "" as! Content,
                rawContent: GeneratedContent(""),
                transcriptEntries: ArraySlice(entries)
            )
        }

        if let toolCalls = choice.message.toolCalls, !toolCalls.isEmpty {
            let invocations = try await resolveToolCalls(toolCalls, session: session)
            if !invocations.isEmpty {
                entries.append(.toolCalls(Transcript.ToolCalls(invocations.map { $0.call })))
                for invocation in invocations {
                    entries.append(.toolOutput(invocation.output))
                }
            }
        }

        let text = choice.message.content ?? ""
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
        let params = Responses.createRequestBody(
            model: model,
            messages: messages,
            tools: tools,
            options: options,
            stream: false
        )

        let url = baseURL.appendingPathComponent("responses")
        let body = try JSONEncoder().encode(params)
        let resp: Responses.Response = try await urlSession.fetch(
            .post,
            url: url,
            headers: [
                "Authorization": "Bearer \(apiKey)"
            ],
            body: body
        )

        var entries: [Transcript.Entry] = []

        if let toolCalls = resp.toolCalls, !toolCalls.isEmpty {
            let invocations = try await resolveToolCalls(toolCalls, session: session)
            if !invocations.isEmpty {
                entries.append(.toolCalls(Transcript.ToolCalls(invocations.map { $0.call })))
                for invocation in invocations {
                    entries.append(.toolOutput(invocation.output))
                }
            }
        }

        let text = resp.outputText ?? ""
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

        let messages = [
            OpenAIMessage(role: .user, content: .text(prompt.description))
        ]

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
                                    "Authorization": "Bearer \(apiKey)"
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
                                    "Authorization": "Bearer \(apiKey)"
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
            body["tools"] = .array(tools.map { $0.jsonValue })
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

        struct Choice: Decodable, Sendable {
            let message: Message
            let finishReason: String?

            private enum CodingKeys: String, CodingKey {
                case message
                case finishReason = "finish_reason"
            }
        }

        struct Message: Decodable, Sendable {
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
        var body: [String: JSONValue] = [
            "model": .string(model),
            "input": .object(["type": .string("input_text"), "text": .string("")]),
            "messages": .array(messages.map { $0.jsonValue(for: .responses) }),
            "stream": .bool(stream),
        ]

        if let tools {
            body["tools"] = .array(tools.map { $0.jsonValue })
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
        let outputText: String?
        let finishReason: String?
        let toolCalls: [OpenAIToolCall]?

        private enum CodingKeys: String, CodingKey {
            case id
            case outputText = "output_text"
            case finishReason = "finish_reason"
            case toolCalls = "tool_calls"
        }
    }
}

// MARK: - Supporting Types

private struct OpenAIMessage: Hashable, Codable, Sendable {
    enum Role: String, Hashable, Codable, Sendable { case system, user, assistant, tool }

    enum Content: Hashable, Codable, Sendable {
        case text(String)
    }

    let role: Role
    let content: Content

    func jsonValue(for apiVariant: OpenAILanguageModel.APIVariant) -> JSONValue {
        switch content {
        case .text(let text):
            switch apiVariant {
            case .chatCompletions:
                // Chat Completions uses simple string content
                return .object([
                    "role": .string(role.rawValue),
                    "content": .string(text),
                ])
            case .responses:
                // Responses API uses array of content blocks
                return .object([
                    "role": .string(role.rawValue),
                    "content": .array([.object(["type": .string("text"), "text": .string(text)])]),
                ])
            }
        }
    }
}

private struct OpenAITool: Hashable, Codable, Sendable {
    let type: String
    let function: OpenAIFunction

    var jsonValue: JSONValue {
        return .object([
            "type": .string(type),
            "function": function.jsonValue,
        ])
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

private struct OpenAIToolCall: Decodable, Sendable {
    let id: String?
    let type: String?
    let function: OpenAIToolFunction?
}

private struct OpenAIToolFunction: Decodable, Sendable {
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
            self = .outputTextDelta(try container.decode(String.self, forKey: .text))
        case "response.tool_call.created":
            self = .toolCallCreated(try container.decode(OpenAIToolCall.self, forKey: .toolCall))
        case "response.tool_call.delta":
            self = .toolCallDelta(try container.decode(OpenAIToolCall.self, forKey: .toolCall))
        case "response.completed":
            self = .completed(try container.decode(String.self, forKey: .finishReason))
        default:
            self = .ignored
        }
    }

    private enum CodingKeys: String, CodingKey {
        case type
        case text
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
    if let raw = try? JSONValue(tool.parameters) {
        rawParameters = raw
    } else {
        rawParameters = nil
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
