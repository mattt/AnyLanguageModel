import Foundation
import Observation

@Observable
public final class LanguageModelSession: @unchecked Sendable {
    public private(set) var isResponding: Bool = false
    public private(set) var transcript: Transcript

    private let model: any LanguageModel
    public let tools: [any Tool]
    public let instructions: Instructions?

    @ObservationIgnored private let respondingState = RespondingState()

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

        // Build transcript with instructions if provided and not already in transcript
        var finalTranscript = transcript
        if let instructions = instructions {
            // Only add instructions if transcript doesn't already start with instructions
            let hasInstructions =
                finalTranscript.first.map { entry in
                    if case .instructions = entry { return true } else { return false }
                } ?? false

            if !hasInstructions {
                let instructionsEntry = Transcript.Entry.instructions(
                    Transcript.Instructions(
                        segments: [.text(.init(content: instructions.description))],
                        toolDefinitions: tools.map { Transcript.ToolDefinition(tool: $0) }
                    )
                )
                finalTranscript.append(instructionsEntry)
            }
        }

        self.transcript = finalTranscript
    }

    public func prewarm(promptPrefix: Prompt? = nil) {
        model.prewarm(for: self, promptPrefix: promptPrefix)
    }

    nonisolated private func beginResponding() async {
        let count = await respondingState.increment()
        let active = count > 0
        await MainActor.run {
            self.isResponding = active
        }
    }

    nonisolated private func endResponding() async {
        let count = await respondingState.decrement()
        let active = count > 0
        await MainActor.run {
            self.isResponding = active
        }
    }

    nonisolated private func wrapRespond<T>(_ operation: () async throws -> T) async throws -> T {
        await beginResponding()
        do {
            let result = try await operation()
            await endResponding()
            return result
        } catch {
            await endResponding()
            throw error
        }
    }

    nonisolated private func wrapStream<Content>(
        _ upstream: sending ResponseStream<Content>
    ) -> ResponseStream<Content> where Content: Generable, Content.PartiallyGenerated: Sendable {
        let session = self
        let relay = AsyncThrowingStream<ResponseStream<Content>.Snapshot, any Error> { continuation in
            let stream = upstream
            Task {
                await session.beginResponding()
                do {
                    for try await snapshot in stream {
                        continuation.yield(snapshot)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
                await session.endResponding()
            }
        }
        return ResponseStream(stream: relay)
    }

    public struct Response<Content>: Sendable where Content: Generable, Content: Sendable {
        public let content: Content
        public let rawContent: GeneratedContent
        public let transcriptEntries: ArraySlice<Transcript.Entry>
    }

    @discardableResult
    nonisolated public func respond(
        to prompt: Prompt,
        options: GenerationOptions = GenerationOptions()
    ) async throws -> Response<String> {
        try await respond(
            to: prompt,
            generating: String.self,
            includeSchemaInPrompt: true,
            options: options
        )
    }

    @discardableResult
    nonisolated public func respond(
        to prompt: String,
        options: GenerationOptions = GenerationOptions()
    ) async throws -> Response<String> {
        try await respond(to: Prompt(prompt), options: options)
    }

    @discardableResult
    nonisolated public func respond(
        options: GenerationOptions = GenerationOptions(),
        @PromptBuilder prompt: () throws -> Prompt
    ) async throws -> Response<String> {
        try await respond(to: try prompt(), options: options)
    }

    @discardableResult
    nonisolated public func respond(
        to prompt: Prompt,
        schema: GenerationSchema,
        includeSchemaInPrompt: Bool = true,
        options: GenerationOptions = GenerationOptions()
    ) async throws -> Response<GeneratedContent> {
        try await respond(
            to: prompt,
            generating: GeneratedContent.self,
            includeSchemaInPrompt: includeSchemaInPrompt,
            options: options
        )
    }

    @discardableResult
    nonisolated public func respond(
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
    nonisolated public func respond(
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
    nonisolated public func respond<Content>(
        to prompt: Prompt,
        generating type: Content.Type = Content.self,
        includeSchemaInPrompt: Bool = true,
        options: GenerationOptions = GenerationOptions()
    ) async throws -> Response<Content> where Content: Generable {
        try await wrapRespond {
            // Add prompt to transcript
            let promptEntry = Transcript.Entry.prompt(
                Transcript.Prompt(
                    segments: [.text(.init(content: prompt.description))],
                    options: options,
                    responseFormat: nil
                )
            )
            await MainActor.run {
                self.transcript.append(promptEntry)
            }

            let response = try await model.respond(
                within: self,
                to: prompt,
                generating: type,
                includeSchemaInPrompt: includeSchemaInPrompt,
                options: options
            )

            // Add response entries to transcript
            await MainActor.run {
                self.transcript.append(contentsOf: response.transcriptEntries)
            }

            return response
        }
    }

    @discardableResult
    nonisolated public func respond<Content>(
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
    nonisolated public func respond<Content>(
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

    nonisolated public func streamResponse(
        to prompt: Prompt,
        schema: GenerationSchema,
        includeSchemaInPrompt: Bool = true,
        options: GenerationOptions = GenerationOptions()
    ) -> sending ResponseStream<GeneratedContent> {
        wrapStream(
            model.streamResponse(
                within: self,
                to: prompt,
                generating: GeneratedContent.self,
                includeSchemaInPrompt: includeSchemaInPrompt,
                options: options
            )
        )
    }

    nonisolated public func streamResponse(
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

    nonisolated public func streamResponse(
        schema: GenerationSchema,
        includeSchemaInPrompt: Bool = true,
        options: GenerationOptions = GenerationOptions(),
        @PromptBuilder prompt: () throws -> Prompt
    ) rethrows -> sending ResponseStream<GeneratedContent> {
        streamResponse(to: try prompt(), schema: schema, includeSchemaInPrompt: includeSchemaInPrompt, options: options)
    }

    nonisolated public func streamResponse<Content>(
        to prompt: Prompt,
        generating type: Content.Type = Content.self,
        includeSchemaInPrompt: Bool = true,
        options: GenerationOptions = GenerationOptions()
    ) -> sending ResponseStream<Content> where Content: Generable {
        wrapStream(
            model.streamResponse(
                within: self,
                to: prompt,
                generating: type,
                includeSchemaInPrompt: includeSchemaInPrompt,
                options: options
            )
        )
    }

    nonisolated public func streamResponse<Content>(
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
        wrapStream(
            model.streamResponse(
                within: self,
                to: prompt,
                generating: String.self,
                includeSchemaInPrompt: true,
                options: options
            )
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

private actor RespondingState {
    private var count = 0

    func increment() -> Int {
        count += 1
        return count
    }

    func decrement() -> Int {
        count = max(0, count - 1)
        return count
    }
}

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
    public struct ResponseStream<Content>: Sendable where Content: Generable, Content.PartiallyGenerated: Sendable {
        private let content: Content
        private let rawContent: GeneratedContent
        private let streaming: AsyncThrowingStream<Snapshot, any Error>?

        init(content: Content, rawContent: GeneratedContent) {
            self.content = content
            self.rawContent = rawContent
            self.streaming = nil
        }

        init(stream: AsyncThrowingStream<Snapshot, any Error>) {
            // Fallback values when consumers call collect() before any snapshots arrive
            // These will be replaced by the last yielded snapshot during collect()
            self.content = (try? Content(GeneratedContent(""))) ?? ("" as! Content)
            self.rawContent = GeneratedContent("")
            self.streaming = stream
        }

        public struct Snapshot: Sendable where Content.PartiallyGenerated: Sendable {
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
        private var streamIterator: AsyncThrowingStream<Snapshot, any Error>.AsyncIterator?
        private let useStream: Bool

        init(content: Content, rawContent: GeneratedContent, stream: AsyncThrowingStream<Snapshot, any Error>?) {
            self.content = content
            self.rawContent = rawContent
            if let stream {
                let iterator = stream.makeAsyncIterator()
                self.streamIterator = iterator
                self.useStream = true
            } else {
                self.streamIterator = nil
                self.useStream = false
            }
        }

        public mutating func next() async throws -> Snapshot? {
            if useStream {
                if var iterator = streamIterator {
                    if let value = try await iterator.next() {
                        // store back the advanced iterator
                        streamIterator = iterator
                        return value
                    }
                    streamIterator = iterator
                }
                return nil
            } else {
                guard !hasYielded else { return nil }
                hasYielded = true
                return Snapshot(
                    content: content.asPartiallyGenerated(),
                    rawContent: rawContent
                )
            }
        }

        public typealias Element = Snapshot
    }

    public func makeAsyncIterator() -> AsyncIterator {
        return AsyncIterator(content: content, rawContent: rawContent, stream: streaming)
    }

    nonisolated public func collect() async throws -> sending LanguageModelSession.Response<Content> {
        if let streaming {
            var last: Snapshot?
            for try await snapshot in streaming {
                last = snapshot
            }
            if let last {
                // Attempt to materialize a concrete Content from the last snapshot
                let finalContent: Content
                if let concrete = last.content as? Content {
                    finalContent = concrete
                } else {
                    finalContent = try Content(last.rawContent)
                }
                return LanguageModelSession.Response(
                    content: finalContent,
                    rawContent: last.rawContent,
                    transcriptEntries: []
                )
            }
        }
        return LanguageModelSession.Response(
            content: content,
            rawContent: rawContent,
            transcriptEntries: []
        )
    }
}
