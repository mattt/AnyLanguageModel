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

    /// Custom generation options specific to Gemini models.
    ///
    /// Use this type to configure Gemini-specific features like thinking mode
    /// and server-side tools through ``GenerationOptions``.
    ///
    /// ```swift
    /// var options = GenerationOptions(temperature: 0.7)
    /// options[custom: GeminiLanguageModel.self] = .init(
    ///     thinking: .dynamic,
    ///     serverTools: [.googleSearch]
    /// )
    /// ```
    public struct CustomGenerationOptions: AnyLanguageModel.CustomGenerationOptions {
        /// Configures thinking (extended reasoning) behavior for Gemini models.
        ///
        /// Use this type to enable or configure thinking mode, which allows the model
        /// to perform extended reasoning before generating a response.
        public enum Thinking: Sendable, Hashable, ExpressibleByBooleanLiteral, ExpressibleByIntegerLiteral {
            /// Thinking is disabled.
            case disabled
            /// Thinking is enabled with dynamic budget allocation.
            case dynamic
            /// Thinking is enabled with a specific token budget.
            case budget(Int)

            var budgetValue: Int? {
                switch self {
                case .disabled: return 0
                case .dynamic: return -1
                case .budget(let value): return value
                }
            }

            public init(booleanLiteral value: Bool) {
                self = value ? .dynamic : .disabled
            }

            public init(integerLiteral value: Int) {
                self = .budget(value)
            }
        }

        /// Server-side tools available for Gemini models.
        ///
        /// These tools are executed by Google's servers and provide access to
        /// external services like search, code execution, and maps.
        public enum ServerTool: Sendable, Hashable {
            /// Google Search for real-time information retrieval.
            case googleSearch
            /// URL context for fetching and analyzing web page content.
            case urlContext
            /// Code execution sandbox for running code snippets.
            case codeExecution
            /// Google Maps for location-based queries.
            /// - Parameters:
            ///   - latitude: Optional latitude for location context.
            ///   - longitude: Optional longitude for location context.
            case googleMaps(latitude: Double?, longitude: Double?)
        }

        /// The thinking mode configuration.
        ///
        /// When set, this enables extended reasoning before the model generates
        /// its response. Use `.dynamic` for automatic budget allocation, or
        /// `.budget(_:)` for a specific token budget.
        public var thinking: Thinking?

        /// Server-side tools to enable for this request.
        ///
        /// These tools are executed by Google's servers and can provide
        /// access to real-time information (Google Search), web content
        /// (URL context), code execution, and location services (Google Maps).
        public var serverTools: [ServerTool]?

        /// Creates custom generation options for Gemini models.
        ///
        /// - Parameters:
        ///   - thinking: The thinking mode configuration. When `nil`, uses the model's default.
        ///   - serverTools: Server-side tools to enable. When `nil`, uses the model's default.
        public init(
            thinking: Thinking? = nil,
            serverTools: [ServerTool]? = nil
        ) {
            self.thinking = thinking
            self.serverTools = serverTools
        }
    }

    /// Deprecated. Use ``CustomGenerationOptions/Thinking`` instead.
    @available(*, deprecated, renamed: "CustomGenerationOptions.Thinking")
    public typealias Thinking = CustomGenerationOptions.Thinking

    /// Deprecated. Use ``CustomGenerationOptions/ServerTool`` instead.
    @available(*, deprecated, renamed: "CustomGenerationOptions.ServerTool")
    public typealias ServerTool = CustomGenerationOptions.ServerTool

    public let baseURL: URL

    private let tokenProvider: @Sendable () -> String

    public let apiVersion: String

    public let model: String

    /// The thinking mode for this model.
    ///
    /// - Important: This property is deprecated. Use ``GenerationOptions`` with
    ///   custom options instead:
    ///   ```swift
    ///   var options = GenerationOptions()
    ///   options[custom: GeminiLanguageModel.self] = .init(thinking: .dynamic)
    ///   ```
    @available(*, deprecated, message: "Use GenerationOptions with custom options instead")
    public var thinking: Thinking {
        get { _thinking }
        set { _thinking = newValue }
    }

    /// Internal storage for the deprecated thinking property.
    internal var _thinking: CustomGenerationOptions.Thinking

    /// Server-side tools enabled for this model.
    ///
    /// - Important: This property is deprecated. Use ``GenerationOptions`` with
    ///   custom options instead:
    ///   ```swift
    ///   var options = GenerationOptions()
    ///   options[custom: GeminiLanguageModel.self] = .init(serverTools: [.googleSearch])
    ///   ```
    @available(*, deprecated, message: "Use GenerationOptions with custom options instead")
    public var serverTools: [CustomGenerationOptions.ServerTool] {
        get { _serverTools }
        set { _serverTools = newValue }
    }

    /// Internal storage for the deprecated serverTools property.
    internal var _serverTools: [CustomGenerationOptions.ServerTool]

    private let urlSession: URLSession

    /// Creates a new Gemini language model.
    ///
    /// - Parameters:
    ///   - baseURL: The base URL for the Gemini API.
    ///   - tokenProvider: A closure that provides the API key.
    ///   - apiVersion: The API version to use.
    ///   - model: The model identifier.
    ///   - session: The URL session for network requests.
    public init(
        baseURL: URL = defaultBaseURL,
        apiKey tokenProvider: @escaping @autoclosure @Sendable () -> String,
        apiVersion: String = defaultAPIVersion,
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
        self.model = model
        self._thinking = .disabled
        self._serverTools = []
        self.urlSession = session
    }

    /// Creates a new Gemini language model with thinking and server tools configuration.
    ///
    /// - Parameters:
    ///   - baseURL: The base URL for the Gemini API.
    ///   - tokenProvider: A closure that provides the API key.
    ///   - apiVersion: The API version to use.
    ///   - model: The model identifier.
    ///   - thinking: The thinking mode configuration.
    ///   - serverTools: Server-side tools to enable.
    ///   - session: The URL session for network requests.
    ///
    /// - Important: This initializer is deprecated. Use the initializer without
    ///   `thinking` and `serverTools` parameters, and pass these options through
    ///   ``GenerationOptions`` instead.
    @available(
        *,
        deprecated,
        message: "Use init without thinking/serverTools and pass them via GenerationOptions custom options"
    )
    public init(
        baseURL: URL = defaultBaseURL,
        apiKey tokenProvider: @escaping @autoclosure @Sendable () -> String,
        apiVersion: String = defaultAPIVersion,
        model: String,
        thinking: CustomGenerationOptions.Thinking = .disabled,
        serverTools: [CustomGenerationOptions.ServerTool] = [],
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
        self._thinking = thinking
        self._serverTools = serverTools
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

        // Extract effective configuration from custom options or fall back to model defaults
        let customOptions = options[custom: GeminiLanguageModel.self]
        let effectiveThinking = customOptions?.thinking ?? _thinking
        let effectiveServerTools = customOptions?.serverTools ?? _serverTools

        let url =
            baseURL
            .appendingPathComponent(apiVersion)
            .appendingPathComponent("models/\(model):generateContent")
        let headers = buildHeaders()

        let userSegments = extractPromptSegments(from: session, fallbackText: prompt.description)
        var contents = [
            GeminiContent(role: .user, parts: convertSegmentsToGeminiParts(userSegments))
        ]

        let geminiTools = try buildTools(from: session.tools, serverTools: effectiveServerTools)

        var allEntries: [Transcript.Entry] = []

        // Multi-turn conversation loop for tool calling
        while true {
            let params = try createGenerateContentParams(
                contents: contents,
                tools: geminiTools,
                options: options,
                thinking: effectiveThinking
            )

            let body = try JSONEncoder().encode(params)

            let response: GeminiGenerateContentResponse = try await urlSession.fetch(
                .post,
                url: url,
                headers: headers,
                body: body
            )

            guard let firstCandidate = response.candidates.first else {
                throw GeminiError.noCandidate
            }

            let functionCalls: [GeminiFunctionCall] =
                firstCandidate.content.parts?.compactMap { part in
                    if case .functionCall(let call) = part { return call }
                    return nil
                } ?? []

            if !functionCalls.isEmpty {
                // Append the model's response with function calls to the conversation
                contents.append(firstCandidate.content)

                // Resolve function calls
                let invocations = try await resolveFunctionCalls(functionCalls, session: session)
                if !invocations.isEmpty {
                    allEntries.append(.toolCalls(Transcript.ToolCalls(invocations.map(\.call))))

                    // Build tool response parts for Gemini
                    var toolParts: [GeminiPart] = []
                    for invocation in invocations {
                        allEntries.append(.toolOutput(invocation.output))

                        // Convert tool output to function response
                        let responseValue = try toJSONValue(invocation.output)
                        toolParts.append(
                            .functionResponse(
                                GeminiFunctionResponse(
                                    name: invocation.call.toolName,
                                    response: responseValue
                                )
                            )
                        )
                    }

                    // Append tool responses to the conversation
                    contents.append(GeminiContent(role: .tool, parts: toolParts))
                }

                // Continue the loop to send the next request with tool results
                continue
            } else {
                // No function calls, extract final text and return
                let text =
                    firstCandidate.content.parts?.compactMap { part -> String? in
                        switch part {
                        case .text(let t): return t.text
                        default: return nil
                        }
                    }.joined() ?? ""

                return LanguageModelSession.Response(
                    content: text as! Content,
                    rawContent: GeneratedContent(text),
                    transcriptEntries: ArraySlice(allEntries)
                )
            }
        }
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

        // Extract effective configuration from custom options or fall back to model defaults
        let customOptions = options[custom: GeminiLanguageModel.self]
        let effectiveThinking = customOptions?.thinking ?? _thinking
        let effectiveServerTools = customOptions?.serverTools ?? _serverTools

        let userSegments = extractPromptSegments(from: session, fallbackText: prompt.description)
        let contents = [
            GeminiContent(role: .user, parts: convertSegmentsToGeminiParts(userSegments))
        ]

        var streamURL =
            baseURL
            .appendingPathComponent(apiVersion)
            .appendingPathComponent("models/\(model):streamGenerateContent")
        streamURL.append(queryItems: [URLQueryItem(name: "alt", value: "sse")])
        let url = streamURL

        let stream: AsyncThrowingStream<LanguageModelSession.ResponseStream<Content>.Snapshot, any Error> = .init {
            continuation in
            let task = Task { @Sendable in
                do {
                    let headers = buildHeaders()

                    let geminiTools = try buildTools(from: session.tools, serverTools: effectiveServerTools)

                    let params = try createGenerateContentParams(
                        contents: contents,
                        tools: geminiTools,
                        options: options,
                        thinking: effectiveThinking
                    )

                    let body = try JSONEncoder().encode(params)

                    let stream: AsyncThrowingStream<GeminiGenerateContentResponse, any Error> =
                        urlSession
                        .fetchEventStream(
                            .post,
                            url: url,
                            headers: headers,
                            body: body
                        )

                    var accumulatedText = ""

                    for try await chunk in stream {
                        guard let candidate = chunk.candidates.first else { continue }

                        if let parts = candidate.content.parts {
                            for part in parts {
                                if case .text(let textPart) = part {
                                    accumulatedText += textPart.text

                                    let raw = GeneratedContent(accumulatedText)
                                    let content: Content.PartiallyGenerated = (accumulatedText as! Content)
                                        .asPartiallyGenerated()
                                    continuation.yield(.init(content: content, rawContent: raw))
                                }
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

    private func buildTools(from tools: [any Tool], serverTools: [CustomGenerationOptions.ServerTool]) throws
        -> [GeminiTool]?
    {
        var geminiTools: [GeminiTool] = []

        if !tools.isEmpty {
            let functionDeclarations: [GeminiFunctionDeclaration] = try tools.map { tool in
                try convertToolToGeminiFormat(tool)
            }
            geminiTools.append(.functionDeclarations(functionDeclarations))
        }

        for serverTool in serverTools {
            switch serverTool {
            case .googleSearch:
                geminiTools.append(.googleSearch)
            case .urlContext:
                geminiTools.append(.urlContext)
            case .codeExecution:
                geminiTools.append(.codeExecution)
            case .googleMaps(let latitude, let longitude):
                geminiTools.append(.googleMaps(latitude: latitude, longitude: longitude))
            }
        }

        return geminiTools.isEmpty ? nil : geminiTools
    }
}

private func createGenerateContentParams(
    contents: [GeminiContent],
    tools: [GeminiTool]?,
    options: GenerationOptions,
    thinking: GeminiLanguageModel.CustomGenerationOptions.Thinking
) throws -> [String: JSONValue] {
    var params: [String: JSONValue] = [
        "contents": try JSONValue(contents)
    ]

    if let tools, !tools.isEmpty {
        params["tools"] = try .array(tools.map { try $0.jsonValue })

        // Add toolConfig if any tool provides one
        for tool in tools {
            if let toolConfig = tool.toolConfigValue {
                params["toolConfig"] = toolConfig
                break
            }
        }
    }

    var generationConfig: [String: JSONValue] = [:]

    if let maxTokens = options.maximumResponseTokens {
        generationConfig["maxOutputTokens"] = .int(maxTokens)
    }

    if let temperature = options.temperature {
        generationConfig["temperature"] = .double(temperature)
    }

    var thinkingConfig: [String: JSONValue] = [:]
    if case .disabled = thinking {
        thinkingConfig["includeThoughts"] = .bool(false)
    } else {
        thinkingConfig["includeThoughts"] = .bool(true)

        if let budget = thinking.budgetValue {
            thinkingConfig["thinkingBudget"] = .int(budget)
        }
    }
    generationConfig["thinkingConfig"] = .object(thinkingConfig)

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

private func convertToolToGeminiFormat(_ tool: any Tool) throws -> GeminiFunctionDeclaration {
    let resolvedSchema = tool.parameters.withResolvedRoot() ?? tool.parameters

    let encoder = JSONEncoder()
    encoder.userInfo[GenerationSchema.omitAdditionalPropertiesKey] = true
    let data = try encoder.encode(resolvedSchema)
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

private func toJSONValue(_ toolOutput: Transcript.ToolOutput) throws -> [String: JSONValue] {
    var result: [String: JSONValue] = [:]

    for segment in toolOutput.segments {
        switch segment {
        case .text(let text):
            result["result"] = .string(text.content)
        case .structure(let structured):
            // For structured segments, encode the content
            let data = try JSONEncoder().encode(structured.content)
            if let jsonString = String(data: data, encoding: .utf8) {
                result["result"] = .string(jsonString)
            }
        case .image:
            // Ignore images in tool outputs for Gemini conversion
            break
        }
    }

    return result
}

private enum GeminiTool: Sendable {
    case functionDeclarations([GeminiFunctionDeclaration])
    case googleSearch
    case urlContext
    case codeExecution
    case googleMaps(latitude: Double?, longitude: Double?)

    var jsonValue: JSONValue {
        get throws {
            switch self {
            case .functionDeclarations(let declarations):
                return .object(["function_declarations": try JSONValue(declarations)])
            case .googleSearch:
                return .object(["google_search": .object([:])])
            case .urlContext:
                return .object(["url_context": .object([:])])
            case .codeExecution:
                return .object(["code_execution": .object([:])])
            case .googleMaps:
                return .object(["google_maps": .object([:])])
            }
        }
    }

    var toolConfigValue: JSONValue? {
        switch self {
        case .googleMaps(let latitude, let longitude):
            guard let lat = latitude, let lng = longitude else { return nil }
            return .object([
                "retrievalConfig": .object([
                    "latLng": .object([
                        "latitude": .double(lat),
                        "longitude": .double(lng),
                    ])
                ])
            ])
        default:
            return nil
        }
    }
}

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
    let parts: [GeminiPart]?
}

private enum GeminiPart: Codable, Sendable {
    case text(GeminiTextPart)
    case functionCall(GeminiFunctionCall)
    case functionResponse(GeminiFunctionResponse)
    case inlineData(GeminiInlineData)
    case fileData(GeminiFileData)

    enum CodingKeys: String, CodingKey {
        case text
        case functionCall
        case functionResponse
        case thoughtSignature
        case inlineData
        case fileData
    }

    init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        if container.contains(.text) {
            let text = try container.decode(String.self, forKey: .text)
            self = .text(GeminiTextPart(text: text))
        } else if container.contains(.functionCall) {
            // Note: thoughtSignature may be present but is ignored
            self = .functionCall(try container.decode(GeminiFunctionCall.self, forKey: .functionCall))
        } else if container.contains(.functionResponse) {
            self = .functionResponse(try container.decode(GeminiFunctionResponse.self, forKey: .functionResponse))
        } else if container.contains(.inlineData) {
            self = .inlineData(try container.decode(GeminiInlineData.self, forKey: .inlineData))
        } else if container.contains(.fileData) {
            self = .fileData(try container.decode(GeminiFileData.self, forKey: .fileData))
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
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .text(let part):
            try container.encode(part.text, forKey: .text)
        case .functionCall(let call):
            try container.encode(call, forKey: .functionCall)
        case .functionResponse(let response):
            try container.encode(response, forKey: .functionResponse)
        case .inlineData(let data):
            try container.encode(data, forKey: .inlineData)
        case .fileData(let data):
            try container.encode(data, forKey: .fileData)
        }
    }
}

private struct GeminiTextPart: Codable, Sendable {
    let text: String
}

private struct GeminiInlineData: Codable, Sendable {
    let mimeType: String
    let data: String

    enum CodingKeys: String, CodingKey {
        case mimeType = "mime_type"
        case data
    }
}

private struct GeminiFileData: Codable, Sendable {
    let fileURI: String

    enum CodingKeys: String, CodingKey {
        case fileURI = "file_uri"
    }
}

private func convertSegmentsToGeminiParts(_ segments: [Transcript.Segment]) -> [GeminiPart] {
    var parts: [GeminiPart] = []
    parts.reserveCapacity(segments.count)
    for segment in segments {
        switch segment {
        case .text(let t):
            parts.append(.text(GeminiTextPart(text: t.content)))
        case .structure(let s):
            parts.append(.text(GeminiTextPart(text: s.content.jsonString)))
        case .image(let img):
            switch img.source {
            case .data(let data, let mime):
                parts.append(.inlineData(GeminiInlineData(mimeType: mime, data: data.base64EncodedString())))
            case .url(let url):
                parts.append(.fileData(GeminiFileData(fileURI: url.absoluteString)))
            }
        }
    }
    return parts
}

private func extractPromptSegments(from session: LanguageModelSession, fallbackText: String) -> [Transcript.Segment] {
    for entry in session.transcript.reversed() {
        if case .prompt(let p) = entry {
            return p.segments
        }
    }
    return [.text(.init(content: fallbackText))]
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
