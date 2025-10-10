import Foundation
import Observation

@Observable
public final class LanguageModelSession {
    public private(set) var isResponding: Bool = false
    public private(set) var transcript: Transcript

    private let model: any LanguageModel
    public let tools: [any Tool]
    public let instructions: Instructions?

    public convenience init(
        model: any LanguageModel,
        tools: [any Tool] = [],
        @InstructionsBuilder instructions: () throws -> Instructions
    ) rethrows {
        try self.init(model: model, tools: tools, instructions: instructions())
    }

    public convenience init(
        model: any LanguageModel,
        tools: [any Tool] = [],
        instructions: String
    ) {
        self.init(model: model, tools: tools, instructions: Instructions(instructions), transcript: Transcript())
    }

    public convenience init(
        model: any LanguageModel,
        tools: [any Tool] = [],
        instructions: Instructions? = nil
    ) {
        self.init(model: model, tools: tools, instructions: instructions, transcript: Transcript())
    }

    public convenience init(
        model: any LanguageModel,
        tools: [any Tool] = [],
        transcript: Transcript
    ) {
        self.init(model: model, tools: tools, instructions: nil, transcript: transcript)
    }

    private init(
        model: any LanguageModel,
        tools: [any Tool],
        instructions: Instructions?,
        transcript: Transcript
    ) {
        self.model = model
        self.tools = tools
        self.instructions = instructions
        self.transcript = transcript
    }

    public func prewarm(promptPrefix: Prompt? = nil) {
        model.prewarm(for: self, promptPrefix: promptPrefix)
    }

    public struct Response<Content> where Content: Generable {
        public let content: Content
        public let rawContent: GeneratedContent
        public let transcriptEntries: ArraySlice<Transcript.Entry>
    }

    @discardableResult
    nonisolated(nonsending) public func respond(
        to prompt: Prompt,
        options: GenerationOptions = GenerationOptions()
    ) async throws -> Response<String> {
        try await model.respond(
            within: self,
            to: prompt,
            generating: String.self,
            includeSchemaInPrompt: true,
            options: options
        )
    }

    @discardableResult
    nonisolated(nonsending) public func respond(
        to prompt: String,
        options: GenerationOptions = GenerationOptions()
    ) async throws -> Response<String> {
        try await respond(to: Prompt(prompt), options: options)
    }

    @discardableResult
    nonisolated(nonsending) public func respond(
        options: GenerationOptions = GenerationOptions(),
        @PromptBuilder prompt: () throws -> Prompt
    ) async throws -> Response<String> {
        try await respond(to: try prompt(), options: options)
    }

    @discardableResult
    nonisolated(nonsending) public func respond(
        to prompt: Prompt,
        schema: GenerationSchema,
        includeSchemaInPrompt: Bool = true,
        options: GenerationOptions = GenerationOptions()
    ) async throws -> Response<GeneratedContent> {
        try await model.respond(
            within: self,
            to: prompt,
            generating: GeneratedContent.self,
            includeSchemaInPrompt: includeSchemaInPrompt,
            options: options
        )
    }

    @discardableResult
    nonisolated(nonsending) public func respond(
        to prompt: String,
        schema: GenerationSchema,
        includeSchemaInPrompt: Bool = true,
        options: GenerationOptions = GenerationOptions()
    ) async throws -> Response<GeneratedContent> {
        try await respond(
            to: Prompt(prompt),
            schema: schema,
            includeSchemaInPrompt: includeSchemaInPrompt,
            options: options
        )
    }

    @discardableResult
    nonisolated(nonsending) public func respond(
        schema: GenerationSchema,
        includeSchemaInPrompt: Bool = true,
        options: GenerationOptions = GenerationOptions(),
        @PromptBuilder prompt: () throws -> Prompt
    ) async throws -> Response<GeneratedContent> {
        try await respond(
            to: try prompt(),
            schema: schema,
            includeSchemaInPrompt: includeSchemaInPrompt,
            options: options
        )
    }

    @discardableResult
    nonisolated(nonsending) public func respond<Content>(
        to prompt: Prompt,
        generating type: Content.Type = Content.self,
        includeSchemaInPrompt: Bool = true,
        options: GenerationOptions = GenerationOptions()
    ) async throws -> Response<Content> where Content: Generable {
        try await model.respond(
            within: self,
            to: prompt,
            generating: type,
            includeSchemaInPrompt: includeSchemaInPrompt,
            options: options
        )
    }

    @discardableResult
    nonisolated(nonsending) public func respond<Content>(
        to prompt: String,
        generating type: Content.Type = Content.self,
        includeSchemaInPrompt: Bool = true,
        options: GenerationOptions = GenerationOptions()
    ) async throws -> Response<Content> where Content: Generable {
        try await respond(
            to: Prompt(prompt),
            generating: type,
            includeSchemaInPrompt: includeSchemaInPrompt,
            options: options
        )
    }

    @discardableResult
    nonisolated(nonsending) public func respond<Content>(
        generating type: Content.Type = Content.self,
        includeSchemaInPrompt: Bool = true,
        options: GenerationOptions = GenerationOptions(),
        @PromptBuilder prompt: () throws -> Prompt
    ) async throws -> Response<Content> where Content: Generable {
        try await respond(
            to: try prompt(),
            generating: type,
            includeSchemaInPrompt: includeSchemaInPrompt,
            options: options
        )
    }

    public func streamResponse(
        to prompt: Prompt,
        schema: GenerationSchema,
        includeSchemaInPrompt: Bool = true,
        options: GenerationOptions = GenerationOptions()
    ) -> sending ResponseStream<GeneratedContent> {
        model.streamResponse(
            within: self,
            to: prompt,
            generating: GeneratedContent.self,
            includeSchemaInPrompt: includeSchemaInPrompt,
            options: options
        )
    }

    public func streamResponse(
        to prompt: String,
        schema: GenerationSchema,
        includeSchemaInPrompt: Bool = true,
        options: GenerationOptions = GenerationOptions()
    ) -> sending ResponseStream<GeneratedContent> {
        streamResponse(
            to: Prompt(prompt),
            schema: schema,
            includeSchemaInPrompt: includeSchemaInPrompt,
            options: options
        )
    }

    public func streamResponse(
        schema: GenerationSchema,
        includeSchemaInPrompt: Bool = true,
        options: GenerationOptions = GenerationOptions(),
        @PromptBuilder prompt: () throws -> Prompt
    ) rethrows -> sending ResponseStream<GeneratedContent> {
        streamResponse(to: try prompt(), schema: schema, includeSchemaInPrompt: includeSchemaInPrompt, options: options)
    }

    public func streamResponse<Content>(
        to prompt: Prompt,
        generating type: Content.Type = Content.self,
        includeSchemaInPrompt: Bool = true,
        options: GenerationOptions = GenerationOptions()
    ) -> sending ResponseStream<Content> where Content: Generable {
        model.streamResponse(
            within: self,
            to: prompt,
            generating: type,
            includeSchemaInPrompt: includeSchemaInPrompt,
            options: options
        )
    }

    public func streamResponse<Content>(
        to prompt: String,
        generating type: Content.Type = Content.self,
        includeSchemaInPrompt: Bool = true,
        options: GenerationOptions = GenerationOptions()
    ) -> sending ResponseStream<Content> where Content: Generable {
        streamResponse(
            to: Prompt(prompt),
            generating: type,
            includeSchemaInPrompt: includeSchemaInPrompt,
            options: options
        )
    }

    public func streamResponse<Content>(
        generating type: Content.Type = Content.self,
        includeSchemaInPrompt: Bool = true,
        options: GenerationOptions = GenerationOptions(),
        @PromptBuilder prompt: () throws -> Prompt
    ) rethrows -> sending ResponseStream<Content> where Content: Generable {
        streamResponse(
            to: try prompt(),
            generating: type,
            includeSchemaInPrompt: includeSchemaInPrompt,
            options: options
        )
    }

    public func streamResponse(
        to prompt: Prompt,
        options: GenerationOptions = GenerationOptions()
    ) -> sending ResponseStream<String> {
        model.streamResponse(
            within: self,
            to: prompt,
            generating: String.self,
            includeSchemaInPrompt: true,
            options: options
        )
    }

    public func streamResponse(
        to prompt: String,
        options: GenerationOptions = GenerationOptions()
    ) -> sending ResponseStream<String> {
        streamResponse(to: Prompt(prompt), options: options)
    }

    public func streamResponse(
        options: GenerationOptions = GenerationOptions(),
        @PromptBuilder prompt: () throws -> Prompt
    ) rethrows -> sending ResponseStream<String> {
        streamResponse(to: try prompt(), options: options)
    }

    @discardableResult
    public func logFeedbackAttachment(
        sentiment: LanguageModelFeedback.Sentiment?,
        issues: [LanguageModelFeedback.Issue] = [],
        desiredOutput: Transcript.Entry? = nil
    ) -> Data {
        model.logFeedbackAttachment(
            within: self,
            sentiment: sentiment,
            issues: issues,
            desiredOutput: desiredOutput
        )
    }
}

extension LanguageModelSession: @unchecked Sendable {}

extension LanguageModelSession: nonisolated Observable {}

extension LanguageModelSession {
    public enum GenerationError: Error, LocalizedError {
        public struct Context: Sendable {
            public let debugDescription: String

            public init(debugDescription: String) {
                self.debugDescription = debugDescription
            }
        }

        public struct Refusal: Sendable {
            public let transcriptEntries: [Transcript.Entry]

            public init(transcriptEntries: [Transcript.Entry]) {
                self.transcriptEntries = transcriptEntries
            }

            public var explanation: Response<String> {
                get async throws {
                    // Extract explanation from transcript entries
                    let explanationText = transcriptEntries.compactMap { entry in
                        if case .response(let response) = entry {
                            return response.segments.compactMap { segment in
                                if case .text(let textSegment) = segment {
                                    return textSegment.content
                                }
                                return nil
                            }.joined(separator: " ")
                        }
                        return nil
                    }.joined(separator: "\n")

                    return Response(
                        content: explanationText.isEmpty ? "No explanation available" : explanationText,
                        rawContent: GeneratedContent(
                            explanationText.isEmpty ? "No explanation available" : explanationText
                        ),
                        transcriptEntries: ArraySlice(transcriptEntries)
                    )
                }
            }

            public var explanationStream: ResponseStream<String> {
                // Create a simple stream that yields the explanation text
                let explanationText = transcriptEntries.compactMap { entry in
                    if case .response(let response) = entry {
                        return response.segments.compactMap { segment in
                            if case .text(let textSegment) = segment {
                                return textSegment.content
                            }
                            return nil
                        }.joined(separator: " ")
                    }
                    return nil
                }.joined(separator: "\n")

                let finalText = explanationText.isEmpty ? "No explanation available" : explanationText
                return ResponseStream(content: finalText, rawContent: GeneratedContent(finalText))
            }
        }

        case exceededContextWindowSize(Context)
        case assetsUnavailable(Context)
        case guardrailViolation(Context)
        case unsupportedGuide(Context)
        case unsupportedLanguageOrLocale(Context)
        case decodingFailure(Context)
        case rateLimited(Context)
        case concurrentRequests(Context)
        case refusal(Refusal, Context)

        public var errorDescription: String? { nil }
        public var recoverySuggestion: String? { nil }
        public var failureReason: String? { nil }
    }

    public struct ToolCallError: Error, LocalizedError {
        public var tool: any Tool
        public var underlyingError: any Error

        public init(tool: any Tool, underlyingError: any Error) {
            self.tool = tool
            self.underlyingError = underlyingError
        }

        public var errorDescription: String? { nil }
    }
}

extension LanguageModelSession {
    public struct ResponseStream<Content> where Content: Generable {
        private let content: Content
        private let rawContent: GeneratedContent

        init(content: Content, rawContent: GeneratedContent) {
            self.content = content
            self.rawContent = rawContent
        }

        public struct Snapshot {
            public var content: Content.PartiallyGenerated
            public var rawContent: GeneratedContent
        }
    }
}

extension LanguageModelSession.ResponseStream: AsyncSequence {
    public typealias Element = Snapshot

    public struct AsyncIterator: AsyncIteratorProtocol {
        private var hasYielded = false
        private let content: Content
        private let rawContent: GeneratedContent

        init(content: Content, rawContent: GeneratedContent) {
            self.content = content
            self.rawContent = rawContent
        }

        public mutating func next() async throws -> Snapshot? {
            guard !hasYielded else { return nil }
            hasYielded = true

            return Snapshot(
                content: content.asPartiallyGenerated(),
                rawContent: rawContent
            )
        }

        public typealias Element = Snapshot
    }

    public func makeAsyncIterator() -> AsyncIterator {
        return AsyncIterator(content: content, rawContent: rawContent)
    }

    nonisolated(nonsending) public func collect() async throws -> sending LanguageModelSession.Response<Content> {
        return LanguageModelSession.Response(
            content: content,
            rawContent: rawContent,
            transcriptEntries: []
        )
    }
}
