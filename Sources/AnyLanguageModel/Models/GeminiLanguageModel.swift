import EventSource
import Foundation
import JSONSchema
import OrderedCollections

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

public struct GeminiLanguageModel: LanguageModel {
    public typealias UnavailableReason = Never

    public static let defaultBaseURL = URL(string: "https://generativelanguage.googleapis.com")!

    public static let defaultAPIVersion = "v1beta"

    public let baseURL: URL

    private let tokenProvider: @Sendable () -> String

    public let apiVersion: String

    public let model: String

    public var thinkingBudget: Int?

    public var includeThoughts: Bool

    public var enableGoogleSearch: Bool

    private let urlSession: URLSession

    public init(
        baseURL: URL = defaultBaseURL,
        apiKey tokenProvider: @escaping @autoclosure @Sendable () -> String,
        apiVersion: String = defaultAPIVersion,
        model: String,
        thinkingBudget: Int? = nil,
        includeThoughts: Bool = false,
        enableGoogleSearch: Bool = false,
        session: URLSession = URLSession(configuration: .default)
    ) {
        var baseURL = baseURL
        if !baseURL.path.hasSuffix("/") {
            baseURL = baseURL.appendingPathComponent("")
        }

        self.baseURL = baseURL
        self.tokenProvider = tokenProvider
        self.apiVersion = apiVersion
        self.model = model
        self.thinkingBudget = thinkingBudget
        self.includeThoughts = includeThoughts
        self.enableGoogleSearch = enableGoogleSearch
        self.urlSession = session
    }

    public func respond<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
        guard type == String.self else {
            fatalError("GeminiLanguageModel only supports generating String content")
        }

        let url =
            baseURL
            .appendingPathComponent(apiVersion)
            .appendingPathComponent("models/\(model):generateContent")
        let headers = buildHeaders()

        let contents = [
            GeminiContent(role: .user, parts: [.text(GeminiTextPart(text: prompt.description))])
        ]

        let geminiTools = try buildTools(from: session.tools)

        let params = try createGenerateContentParams(
            contents: contents,
            tools: geminiTools,
            options: options,
            thinkingBudget: thinkingBudget,
            includeThoughts: includeThoughts
        )

        let body = try JSONEncoder().encode(params)

        let response: GeminiGenerateContentResponse = try await urlSession.fetch(
            .post,
            url: url,
            headers: headers,
            body: body
        )

        var entries: [Transcript.Entry] = []

        guard let firstCandidate = response.candidates.first else {
            throw GeminiError.noCandidate
        }

        let functionCalls: [GeminiFunctionCall] = firstCandidate.content.parts.compactMap { part in
            if case .functionCall(let call) = part { return call }
            return nil
        }

        if !functionCalls.isEmpty {
            let invocations = try await resolveFunctionCalls(functionCalls, session: session)
            if !invocations.isEmpty {
                entries.append(.toolCalls(Transcript.ToolCalls(invocations.map(\.call))))
                for invocation in invocations {
                    entries.append(.toolOutput(invocation.output))
                }
            }
        }

        let text = firstCandidate.content.parts.compactMap { part -> String? in
            switch part {
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
        guard type == String.self else {
            fatalError("GeminiLanguageModel only supports generating String content")
        }

        let contents = [
            GeminiContent(role: .user, parts: [.text(GeminiTextPart(text: prompt.description))])
        ]

        let url =
            baseURL
            .appendingPathComponent(apiVersion)
            .appendingPathComponent("models/\(model):streamGenerateContent")

        let thinkingBudget = self.thinkingBudget
        let includeThoughts = self.includeThoughts

        let stream: AsyncThrowingStream<LanguageModelSession.ResponseStream<Content>.Snapshot, any Error> = .init {
            continuation in
            let task = Task { @Sendable in
                do {
                    let headers = buildHeaders()

                    let geminiTools = try buildTools(from: session.tools)

                    var params = try createGenerateContentParams(
                        contents: contents,
                        tools: geminiTools,
                        options: options,
                        thinkingBudget: thinkingBudget,
                        includeThoughts: includeThoughts
                    )
                    params["stream"] = .bool(true)

                    let body = try JSONEncoder().encode(params)

                    let stream: AsyncThrowingStream<GeminiGenerateContentResponse, any Error> =
                        urlSession
                        .fetchStream(
                            .post,
                            url: url,
                            headers: headers,
                            body: body
                        )

                    var accumulatedText = ""

                    for try await chunk in stream {
                        guard let candidate = chunk.candidates.first else { continue }

                        for part in candidate.content.parts {
                            if case .text(let textPart) = part {
                                accumulatedText += textPart.text

                                let raw = GeneratedContent(accumulatedText)
                                let content: Content.PartiallyGenerated = (accumulatedText as! Content)
                                    .asPartiallyGenerated()
                                continuation.yield(.init(content: content, rawContent: raw))
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

    private func buildHeaders() -> [String: String] {
        let headers: [String: String] = [
            "x-goog-api-key": tokenProvider()
        ]

        return headers
    }

    private func buildTools(from tools: [any Tool]) throws -> [GeminiTool]? {
        var geminiTools: [GeminiTool] = []

        if !tools.isEmpty {
            let functionDeclarations: [GeminiFunctionDeclaration] = try tools.map { tool in
                try convertToolToGeminiFormat(tool)
            }
            geminiTools.append(GeminiTool(functionDeclarations: functionDeclarations))
        }

        if enableGoogleSearch {
            geminiTools.append(GeminiTool(googleSearch: GeminiGoogleSearch()))
        }

        return geminiTools.isEmpty ? nil : geminiTools
    }
}

private func createGenerateContentParams(
    contents: [GeminiContent],
    tools: [GeminiTool]?,
    options: GenerationOptions,
    thinkingBudget: Int?,
    includeThoughts: Bool
) throws -> [String: JSONValue] {
    var params: [String: JSONValue] = [
        "contents": try JSONValue(contents)
    ]

    if let tools, !tools.isEmpty {
        params["tools"] = try JSONValue(tools)
    }

    var generationConfig: [String: JSONValue] = [:]

    if let maxTokens = options.maximumResponseTokens {
        generationConfig["maxOutputTokens"] = .int(maxTokens)
    }

    if let temperature = options.temperature {
        generationConfig["temperature"] = .double(temperature)
    }

    if thinkingBudget != nil || includeThoughts {
        var thinkingConfig: [String: JSONValue] = [:]

        if let budget = thinkingBudget {
            thinkingConfig["thinkingBudget"] = .int(budget)
        }

        if includeThoughts {
            thinkingConfig["includeThoughts"] = .bool(true)
        }

        if !thinkingConfig.isEmpty {
            generationConfig["thinkingConfig"] = .object(thinkingConfig)
        }
    }

    if !generationConfig.isEmpty {
        params["generationConfig"] = .object(generationConfig)
    }

    return params
}

private struct ToolInvocationResult {
    let call: Transcript.ToolCall
    let output: Transcript.ToolOutput
}

private func resolveFunctionCalls(
    _ functionCalls: [GeminiFunctionCall],
    session: LanguageModelSession
) async throws -> [ToolInvocationResult] {
    if functionCalls.isEmpty { return [] }

    var toolsByName: [String: any Tool] = [:]
    for tool in session.tools {
        if toolsByName[tool.name] == nil {
            toolsByName[tool.name] = tool
        }
    }

    var results: [ToolInvocationResult] = []
    results.reserveCapacity(functionCalls.count)

    for call in functionCalls {
        let args = try toGeneratedContent(call.args)
        let callID = UUID().uuidString
        let transcriptCall = Transcript.ToolCall(
            id: callID,
            toolName: call.name,
            arguments: args
        )

        guard let tool = toolsByName[call.name] else {
            let message = Transcript.Segment.text(.init(content: "Tool not found: \(call.name)"))
            let output = Transcript.ToolOutput(
                id: callID,
                toolName: call.name,
                segments: [message]
            )
            results.append(ToolInvocationResult(call: transcriptCall, output: output))
            continue
        }

        do {
            let segments = try await tool.makeOutputSegments(from: args)
            let output = Transcript.ToolOutput(
                id: callID,
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

private func convertToolToGeminiFormat(_ tool: any Tool) throws -> GeminiFunctionDeclaration {
    let resolvedSchema = tool.parameters.withResolvedRoot() ?? tool.parameters

    let data = try JSONEncoder().encode(resolvedSchema)
    let schema = try JSONDecoder().decode(JSONSchema.self, from: data)

    return GeminiFunctionDeclaration(
        name: tool.name,
        description: tool.description,
        parameters: schema
    )
}

private func toGeneratedContent(_ value: [String: JSONValue]?) throws -> GeneratedContent {
    guard let value else { return GeneratedContent(properties: [:]) }
    let data = try JSONEncoder().encode(JSONValue.object(value))
    let json = String(data: data, encoding: .utf8) ?? "{}"
    return try GeneratedContent(json: json)
}

private struct GeminiTool: Codable, Sendable {
    let functionDeclarations: [GeminiFunctionDeclaration]?
    let googleSearch: GeminiGoogleSearch?

    enum CodingKeys: String, CodingKey {
        case functionDeclarations = "function_declarations"
        case googleSearch = "google_search"
    }

    init(functionDeclarations: [GeminiFunctionDeclaration]) {
        self.functionDeclarations = functionDeclarations
        self.googleSearch = nil
    }

    init(googleSearch: GeminiGoogleSearch) {
        self.functionDeclarations = nil
        self.googleSearch = googleSearch
    }
}

private struct GeminiGoogleSearch: Codable, Sendable {}

private struct GeminiFunctionDeclaration: Codable, Sendable {
    let name: String
    let description: String
    let parameters: JSONSchema
}

private struct GeminiContent: Codable, Sendable {
    enum Role: String, Codable, Sendable {
        case user
        case model
        case tool
    }

    let role: Role
    let parts: [GeminiPart]
}

private enum GeminiPart: Codable, Sendable {
    case text(GeminiTextPart)
    case functionCall(GeminiFunctionCall)
    case functionResponse(GeminiFunctionResponse)

    enum CodingKeys: String, CodingKey {
        case text
        case functionCall
        case functionResponse
    }

    init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        if container.contains(.text) {
            self = .text(try GeminiTextPart(from: decoder))
        } else if container.contains(.functionCall) {
            self = .functionCall(try GeminiFunctionCall(from: decoder))
        } else if container.contains(.functionResponse) {
            self = .functionResponse(try GeminiFunctionResponse(from: decoder))
        } else {
            throw DecodingError.dataCorrupted(
                DecodingError.Context(
                    codingPath: decoder.codingPath,
                    debugDescription: "Unable to decode GeminiPart"
                )
            )
        }
    }

    func encode(to encoder: any Encoder) throws {
        switch self {
        case .text(let part): try part.encode(to: encoder)
        case .functionCall(let call): try call.encode(to: encoder)
        case .functionResponse(let response): try response.encode(to: encoder)
        }
    }
}

private struct GeminiTextPart: Codable, Sendable {
    let text: String
}

private struct GeminiFunctionCall: Codable, Sendable {
    let name: String
    let args: [String: JSONValue]?

    enum CodingKeys: String, CodingKey {
        case name
        case args
    }
}

private struct GeminiFunctionResponse: Codable, Sendable {
    let name: String
    let response: [String: JSONValue]
}

private struct GeminiGenerateContentResponse: Codable, Sendable {
    let candidates: [GeminiCandidate]
    let usageMetadata: GeminiUsageMetadata?

    enum CodingKeys: String, CodingKey {
        case candidates
        case usageMetadata = "usageMetadata"
    }
}

private struct GeminiCandidate: Codable, Sendable {
    let content: GeminiContent
    let finishReason: String?

    enum CodingKeys: String, CodingKey {
        case content
        case finishReason
    }
}

private struct GeminiUsageMetadata: Codable, Sendable {
    let promptTokenCount: Int?
    let candidatesTokenCount: Int?
    let totalTokenCount: Int?
    let thoughtsTokenCount: Int?

    enum CodingKeys: String, CodingKey {
        case promptTokenCount
        case candidatesTokenCount
        case totalTokenCount
        case thoughtsTokenCount
    }
}

enum GeminiError: Error, CustomStringConvertible {
    case noCandidate

    var description: String {
        switch self {
        case .noCandidate:
            return "No candidate in response"
        }
    }
}
