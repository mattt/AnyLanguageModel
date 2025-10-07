import Foundation
import JSONSchema
import OrderedCollections

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

public struct AnthropicLanguageModel: LanguageModel {
    public static let defaultBaseURL = URL(string: "https://api.anthropic.com")!
    public static let defaultAPIVersion = "2023-06-01"

    public let baseURL: URL
    public let apiKey: String
    public let model: String
    private let urlSession: URLSession

    public init(
        baseURL: URL = defaultBaseURL,
        apiKey: String,
        model: String,
        session: URLSession = URLSession(configuration: .default)
    ) {
        var baseURL = baseURL
        if !baseURL.path.hasSuffix("/") {
            baseURL = baseURL.appendingPathComponent("")
        }

        self.baseURL = baseURL
        self.apiKey = apiKey
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

        let messages = [
            AnthropicMessage(role: .user, content: [.text(.init(text: prompt.description))])
        ]
        let params = try createMessageParams(
            model: model,
            system: nil,
            messages: messages,
            tools: nil,
            options: options
        )

        let url = baseURL.appendingPathComponent("v1/messages")
        let body = try JSONEncoder().encode(params)
        let message: AnthropicMessageResponse = try await urlSession.fetch(
            .post,
            url: url,
            headers: [
                "x-api-key": apiKey,
                "anthropic-version": Self.defaultAPIVersion,
            ],
            body: body
        )

        let text = message.content.compactMap { block -> String? in
            switch block {
            case .text(let t): return t.text
            default: return nil
            }
        }.joined()

        return LanguageModelSession.Response(
            content: text as! Content,
            rawContent: GeneratedContent(text),
            transcriptEntries: []
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

        // For AnthropicLanguageModel, we'll simulate streaming by yielding the response immediately
        // In a real implementation, this would make a streaming request to Anthropic's API
        // and yield chunks as they arrive
        // Since we can't make this function async, we'll create a placeholder stream
        let placeholderText = "Anthropic streaming response"
        let generatedContent = GeneratedContent(placeholderText)

        return LanguageModelSession.ResponseStream(content: placeholderText as! Content, rawContent: generatedContent)
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
