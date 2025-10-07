import Foundation
import JSONSchema
import OrderedCollections

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

public struct OllamaLanguageModel: LanguageModel {
    public static let defaultBaseURL = URL(string: "http://localhost:11434")!

    public let baseURL: URL
    public let model: String
    private let urlSession: URLSession

    public init(
        baseURL: URL = defaultBaseURL,
        model: String,
        session: URLSession = URLSession(configuration: .default)
    ) {
        var baseURL = baseURL
        if !baseURL.path.hasSuffix("/") {
            baseURL = baseURL.appendingPathComponent("")
        }

        self.baseURL = baseURL
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
            fatalError("OllamaLanguageModel only supports generating String content")
        }

        let messages = [
            OllamaMessage(role: .user, content: prompt.description)
        ]
        let ollamaOptions = convertOptions(options)
        let ollamaTools = try session.tools.map { tool in
            try convertToolToOllamaFormat(tool)
        }

        let params = try createChatParams(
            model: model,
            messages: messages,
            tools: ollamaTools.isEmpty ? nil : ollamaTools,
            options: ollamaOptions,
            stream: false
        )

        let url = baseURL.appendingPathComponent("api/chat")
        let body = try JSONEncoder().encode(params)
        let chatResponse: ChatResponse = try await urlSession.fetch(
            .post,
            url: url,
            body: body,
            dateDecodingStrategy: .iso8601WithFractionalSeconds
        )

        var entries: [Transcript.Entry] = []

        if let toolCalls = chatResponse.message.toolCalls, !toolCalls.isEmpty {
            // Convert to Transcript.ToolCalls
            let transcriptCalls = try Transcript.ToolCalls(
                toolCalls.map { call in
                    let args = try toGeneratedContent(call.function.arguments)
                    return Transcript.ToolCall(
                        id: call.id ?? UUID().uuidString,
                        toolName: call.function.name,
                        arguments: args
                    )
                }
            )
            entries.append(.toolCalls(transcriptCalls))

            // Execute tools
            for call in toolCalls {
                guard let tool = session.tools.first(where: { $0.name == call.function.name }) else {
                    let segs = [Transcript.Segment.text(.init(content: "Tool not found: \(call.function.name)"))]
                    entries.append(
                        .toolOutput(.init(id: call.function.name, toolName: call.function.name, segments: segs))
                    )
                    continue
                }

                do {
                    let args = try toGeneratedContent(call.function.arguments)
                    let segs = try await tool._invokeErased(with: args)
                    entries.append(.toolOutput(.init(id: tool.name, toolName: tool.name, segments: segs)))
                } catch {
                    throw LanguageModelSession.ToolCallError(tool: tool, underlyingError: error)
                }
            }
        }

        let text = chatResponse.message.content ?? ""
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
            fatalError("OllamaLanguageModel only supports generating String content")
        }

        // For OllamaLanguageModel, we'll simulate streaming by yielding the response immediately
        // In a real implementation, this would make a streaming request to Ollama's API
        // and yield chunks as they arrive
        // Since we can't make this function async, we'll create a placeholder stream
        let placeholderText = "Ollama streaming response"
        let generatedContent = GeneratedContent(placeholderText)

        return LanguageModelSession.ResponseStream(content: placeholderText as! Content, rawContent: generatedContent)
    }
}

// MARK: - Conversions

private func convertOptions(_ options: GenerationOptions) -> [String: JSONValue]? {
    var ollamaOptions: [String: JSONValue] = [:]

    if let temperature = options.temperature {
        ollamaOptions["temperature"] = .double(temperature)
    }
    if let maxTokens = options.maximumResponseTokens {
        ollamaOptions["num_predict"] = .int(maxTokens)
    }

    return ollamaOptions.isEmpty ? nil : ollamaOptions
}

private func convertToolToOllamaFormat(_ tool: any Tool) throws -> [String: JSONValue] {
    return [
        "type": .string("function"),
        "function": .object([
            "name": .string(tool.name),
            "description": .string(tool.description),
            "parameters": try JSONValue(tool.parameters),
        ]),
    ]
}

private func toGeneratedContent(_ value: JSONValue?) throws -> GeneratedContent {
    guard let value else { return GeneratedContent(properties: [:]) }
    let data = try JSONEncoder().encode(value)
    let json = String(data: data, encoding: .utf8) ?? "{}"
    return try GeneratedContent(json: json)
}

private func createChatParams(
    model: String,
    messages: [OllamaMessage],
    tools: [[String: JSONValue]]?,
    options: [String: JSONValue]?,
    stream: Bool
) throws -> [String: JSONValue] {
    var params: [String: JSONValue] = [
        "model": .string(model),
        "messages": try JSONValue(messages),
        "stream": .bool(stream),
    ]

    if let tools {
        params["tools"] = try JSONValue(tools)
    }

    if let options {
        params["options"] = .object(options)
    }

    return params
}

// MARK: - Supporting Types

private struct OllamaMessage: Hashable, Codable, Sendable {
    enum Role: String, Hashable, Codable, Sendable {
        case system
        case user
        case assistant
        case tool
    }

    let role: Role
    let content: String
}

private struct ChatResponse: Decodable, Sendable {
    let model: String
    let createdAt: Date
    let message: ChatMessageResponse
    let done: Bool

    private enum CodingKeys: String, CodingKey {
        case model
        case createdAt = "created_at"
        case message
        case done
    }
}

private struct ChatMessageResponse: Decodable, Sendable {
    let role: OllamaMessage.Role
    let content: String?
    let toolCalls: [OllamaToolCall]?

    private enum CodingKeys: String, CodingKey {
        case role
        case content
        case toolCalls = "tool_calls"
    }
}

private struct OllamaToolCall: Decodable, Sendable {
    let id: String?
    let type: String?
    let function: OllamaToolFunction
}

private struct OllamaToolFunction: Decodable, Sendable {
    let name: String
    let arguments: JSONValue?

    private enum CodingKeys: String, CodingKey {
        case name
        case arguments
    }
}

private struct ErrorResponse: Decodable {
    let error: String
}

extension Tool {
    fileprivate func _invokeErased(with args: GeneratedContent) async throws -> [Transcript.Segment] {
        let parsed = try Arguments(args)
        let output = try await call(arguments: parsed)

        // Convert output to string safely
        let text: String
        if let stringOutput = output as? String {
            text = stringOutput
        } else {
            text = output.promptRepresentation.description
        }

        return [Transcript.Segment.text(.init(content: text))]
    }
}
