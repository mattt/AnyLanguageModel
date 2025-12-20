import Foundation

#if canImport(UIKit)
    import UIKit
    import CoreImage
#endif

#if canImport(AppKit)
    import AppKit
    import CoreImage
#endif

#if MLX
    import MLXLMCommon
    import MLX
    import MLXVLM
    import Tokenizers
    import Hub

    /// Wrapper to store ModelContext in NSCache (requires NSObject subclass).
    private final class CachedContext: NSObject, @unchecked Sendable {
        let context: ModelContext
        init(_ context: ModelContext) { self.context = context }
    }

    /// Coordinates a bounded in-memory cache with structured, coalesced loading.
    private final class ModelContextCache {
        private let cache: NSCache<NSString, CachedContext>
        private let lock = NSLock()
        private var inFlight: [String: Task<CachedContext, Error>] = [:]

        /// Creates a cache with a count-based eviction limit.
        init(countLimit: Int) {
            let cache = NSCache<NSString, CachedContext>()
            cache.countLimit = countLimit
            self.cache = cache
        }

        /// Returns a cached context or loads it exactly once per key.
        func context(
            for key: String,
            loader: @escaping @Sendable () async throws -> ModelContext
        ) async throws -> ModelContext {
            let cacheKey = key as NSString
            if let cached = cache.object(forKey: cacheKey) {
                return cached.context
            }

            if let task = inFlightTask(for: key) {
                return try await task.value.context
            }

            let task = Task { try await CachedContext(loader()) }
            setInFlight(task, for: key)

            do {
                let cached = try await task.value
                cache.setObject(cached, forKey: cacheKey)
                clearInFlight(for: key)
                return cached.context
            } catch {
                clearInFlight(for: key)
                throw error
            }
        }

        /// Removes a cached context for the key.
        func remove(for key: String) {
            cache.removeObject(forKey: key as NSString)
        }

        /// Clears all cached contexts.
        func removeAll() {
            cache.removeAllObjects()
        }

        /// Cancels in-flight work and removes cached data for the key.
        func removeAndCancel(for key: String) async {
            let task = removeInFlight(for: key)
            task?.cancel()
            cache.removeObject(forKey: key as NSString)
        }

        /// Cancels all in-flight work and clears cached data.
        func removeAllAndCancel() async {
            let tasks = removeAllInFlight()
            tasks.forEach { $0.cancel() }
            cache.removeAllObjects()
        }

        private func inFlightTask(for key: String) -> Task<CachedContext, Error>? {
            lock.lock()
            defer { lock.unlock() }
            return inFlight[key]
        }

        private func setInFlight(_ task: Task<CachedContext, Error>, for key: String) {
            lock.lock()
            inFlight[key] = task
            lock.unlock()
        }

        private func clearInFlight(for key: String) {
            lock.lock()
            inFlight[key] = nil
            lock.unlock()
        }

        private func removeInFlight(for key: String) -> Task<CachedContext, Error>? {
            lock.lock()
            defer { lock.unlock() }
            let task = inFlight[key]
            inFlight[key] = nil
            return task
        }

        private func removeAllInFlight() -> [Task<CachedContext, Error>] {
            lock.lock()
            defer { lock.unlock() }
            let tasks = Array(inFlight.values)
            inFlight.removeAll()
            return tasks
        }
    }

    /// Shared cache across MLXLanguageModel instances.
    private nonisolated(unsafe) let modelCache = ModelContextCache(countLimit: 3)

    // MARK: - MLXLanguageModel

    /// A language model that runs locally using MLX.
    ///
    /// Use this model to run language models on Apple silicon using the MLX framework.
    /// Models are automatically downloaded and cached when first used.
    ///
    /// ```swift
    /// let model = MLXLanguageModel(modelId: "mlx-community/Llama-3.2-3B-Instruct-4bit")
    /// ```
    public struct MLXLanguageModel: LanguageModel {
        /// The reason the model is unavailable.
        /// This model is always available.
        public typealias UnavailableReason = Never

        /// The model identifier.
        public let modelId: String

        /// The Hub API instance for downloading models.
        public let hub: HubApi?

        /// The local directory containing the model files.
        public let directory: URL?

        /// Creates an MLX language model.
        ///
        /// - Parameters:
        ///   - modelId: The model identifier (for example, "mlx-community/Llama-3.2-3B-Instruct-4bit").
        ///   - hub: An optional Hub API instance for downloading models. If not provided, the default Hub API is used.
        ///   - directory: An optional local directory URL containing the model files. If provided, the model is loaded from this directory instead of downloading.
        public init(modelId: String, hub: HubApi? = nil, directory: URL? = nil) {
            self.modelId = modelId
            self.hub = hub
            self.directory = directory
        }

        /// Removes this model from the shared cache and cancels any in-flight load.
        ///
        /// Call this to free memory when the model is no longer needed.
        /// The model will be reloaded automatically on the next request.
        public func removeFromCache() async {
            let key = directory?.absoluteString ?? modelId
            await modelCache.removeAndCancel(for: key)
        }

        /// Removes all MLX models from the shared cache and cancels in-flight loads.
        public static func removeAllFromCache() async {
            await modelCache.removeAllAndCancel()
        }

        /// Get or load model context with caching
        private func loadContext(modelId: String, hub: HubApi?, directory: URL?) async throws -> ModelContext {
            let key = directory?.absoluteString ?? modelId

            return try await modelCache.context(for: key) {
                if let directory {
                    return try await loadModel(directory: directory)
                }

                return try await loadModel(hub: hub ?? HubApi(), id: modelId)
            }
        }

        public func respond<Content>(
            within session: LanguageModelSession,
            to prompt: Prompt,
            generating type: Content.Type,
            includeSchemaInPrompt: Bool,
            options: GenerationOptions
        ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
            // Get cached or load fresh ModelContext
            let context = try await loadContext(modelId: modelId, hub: hub, directory: directory)

            if type != String.self {
                let jsonString = try await generateStructuredJSON(
                    session: session,
                    prompt: prompt,
                    context: context,
                    options: options,
                    schema: type.generationSchema
                )

                let generatedContent = try GeneratedContent(json: jsonString)
                let content = try type.init(generatedContent)

                return LanguageModelSession.Response(
                    content: content,
                    rawContent: generatedContent,
                    transcriptEntries: ArraySlice([])
                )
            }

            // Convert session tools to MLX ToolSpec format
            let toolSpecs: [ToolSpec]? =
                session.tools.isEmpty
                ? nil
                : session.tools.map { tool in
                    convertToolToMLXSpec(tool)
                }

            // Map AnyLanguageModel GenerationOptions to MLX GenerateParameters
            let generateParameters = toGenerateParameters(options)

            // Build chat history from full transcript
            var chat = convertTranscriptToMLXChat(session: session, fallbackPrompt: prompt.description)

            var allTextChunks: [String] = []
            var allEntries: [Transcript.Entry] = []

            // Loop until no more tool calls
            while true {
                // Build user input with current chat history and tools
                let userInput = MLXLMCommon.UserInput(
                    chat: chat,
                    processing: .init(resize: .init(width: 512, height: 512)),
                    tools: toolSpecs,
                )
                let lmInput = try await context.processor.prepare(input: userInput)

                // Generate
                let stream = try MLXLMCommon.generate(
                    input: lmInput,
                    parameters: generateParameters,
                    context: context
                )

                var chunks: [String] = []
                var collectedToolCalls: [MLXLMCommon.ToolCall] = []

                for await item in stream {
                    switch item {
                    case .chunk(let text):
                        chunks.append(text)
                    case .info:
                        break
                    case .toolCall(let call):
                        collectedToolCalls.append(call)
                    }
                }

                let assistantText = chunks.joined()
                allTextChunks.append(assistantText)

                // Add assistant response to chat history
                if !assistantText.isEmpty {
                    chat.append(.assistant(assistantText))
                }

                // If there are tool calls, execute them and continue
                if !collectedToolCalls.isEmpty {
                    let invocations = try await resolveToolCalls(collectedToolCalls, session: session)
                    if !invocations.isEmpty {
                        allEntries.append(.toolCalls(Transcript.ToolCalls(invocations.map(\.call))))

                        // Execute each tool and add results to chat
                        for invocation in invocations {
                            allEntries.append(.toolOutput(invocation.output))

                            // Convert tool output to JSON string for MLX
                            let toolResultJSON = toolOutputToJSON(invocation.output)
                            chat.append(.tool(toolResultJSON))
                        }

                        // Continue loop to generate with tool results
                        continue
                    }
                }

                // No more tool calls, exit loop
                break
            }

            let text = allTextChunks.joined()
            return LanguageModelSession.Response(
                content: text as! Content,
                rawContent: GeneratedContent(text),
                transcriptEntries: ArraySlice(allEntries)
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
                fatalError("MLXLanguageModel streaming only supports String content")
            }

            let modelId = self.modelId
            let hub = self.hub
            let directory = self.directory

            let stream: AsyncThrowingStream<LanguageModelSession.ResponseStream<Content>.Snapshot, any Error> = .init {
                continuation in
                let task = Task { @Sendable in
                    do {
                        // Get cached or load fresh ModelContext
                        let context = try await loadContext(modelId: modelId, hub: hub, directory: directory)

                        // Build chat inside task to avoid Sendable issues
                        let generateParameters = toGenerateParameters(options)
                        let chat = convertTranscriptToMLXChat(session: session, fallbackPrompt: prompt.description)

                        let userInput = MLXLMCommon.UserInput(
                            chat: chat,
                            processing: .init(resize: .init(width: 512, height: 512)),
                            tools: nil
                        )
                        let lmInput = try await context.processor.prepare(input: userInput)

                        let mlxStream = try MLXLMCommon.generate(
                            input: lmInput,
                            parameters: generateParameters,
                            context: context
                        )

                        var accumulatedText = ""
                        for await item in mlxStream {
                            if Task.isCancelled { break }

                            switch item {
                            case .chunk(let text):
                                accumulatedText += text
                                let raw = GeneratedContent(accumulatedText)
                                let content: Content.PartiallyGenerated = (accumulatedText as! Content)
                                    .asPartiallyGenerated()
                                continuation.yield(.init(content: content, rawContent: raw))
                            case .info, .toolCall:
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
    }

    // MARK: - Options Mapping

    private func toGenerateParameters(_ options: GenerationOptions) -> MLXLMCommon.GenerateParameters {
        MLXLMCommon.GenerateParameters(
            maxTokens: options.maximumResponseTokens,
            maxKVSize: nil,
            kvBits: nil,
            kvGroupSize: 64,
            quantizedKVStart: 0,
            temperature: Float(options.temperature ?? 0.6),
            topP: 1.0,
            repetitionPenalty: nil,
            repetitionContextSize: 20
        )
    }

    // MARK: - Transcript Conversion

    private func convertTranscriptToMLXChat(
        session: LanguageModelSession,
        fallbackPrompt: String
    ) -> [MLXLMCommon.Chat.Message] {
        var chat: [MLXLMCommon.Chat.Message] = []

        // Check if instructions are already in transcript
        let hasInstructionsInTranscript = session.transcript.contains {
            if case .instructions = $0 { return true }
            return false
        }

        // Add instructions from session if present and not in transcript
        if !hasInstructionsInTranscript,
            let instructions = session.instructions?.description,
            !instructions.isEmpty
        {
            chat.append(.init(role: .system, content: instructions))
        }

        // Convert each transcript entry
        for entry in session.transcript {
            switch entry {
            case .instructions(let instr):
                chat.append(makeMLXChatMessage(from: instr.segments, role: .system))

            case .prompt(let prompt):
                chat.append(makeMLXChatMessage(from: prompt.segments, role: .user))

            case .response(let response):
                let content = response.segments.map { extractText(from: $0) }.joined(separator: "\n")
                chat.append(.assistant(content))

            case .toolCalls:
                // Tool calls are handled inline during generation loop
                break

            case .toolOutput(let toolOutput):
                let content = toolOutput.segments.map { extractText(from: $0) }.joined(separator: "\n")
                chat.append(.tool(content))
            }
        }

        // If no user message in transcript, add fallback prompt
        let hasUserMessage = chat.contains { $0.role == .user }
        if !hasUserMessage {
            chat.append(.init(role: .user, content: fallbackPrompt))
        }

        return chat
    }

    private func extractText(from segment: Transcript.Segment) -> String {
        switch segment {
        case .text(let text):
            return text.content
        case .structure(let structured):
            return structured.content.jsonString
        case .image:
            return ""
        }
    }

    private func makeMLXChatMessage(
        from segments: [Transcript.Segment],
        role: MLXLMCommon.Chat.Message.Role
    ) -> MLXLMCommon.Chat.Message {
        var textParts: [String] = []
        var images: [MLXLMCommon.UserInput.Image] = []

        for segment in segments {
            switch segment {
            case .image(let imageSegment):
                switch imageSegment.source {
                case .url(let url):
                    images.append(.url(url))
                case .data(let data, _):
                    #if canImport(UIKit)
                        if let uiImage = UIKit.UIImage(data: data),
                            let ciImage = CIImage(image: uiImage)
                        {
                            images.append(.ciImage(ciImage))
                        }
                    #elseif canImport(AppKit)
                        if let nsImage = AppKit.NSImage(data: data),
                            let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil)
                        {
                            let ciImage = CIImage(cgImage: cgImage)
                            images.append(.ciImage(ciImage))
                        }
                    #endif
                }
            default:
                let text = extractText(from: segment)
                if !text.isEmpty {
                    textParts.append(text)
                }
            }
        }

        let content = textParts.joined(separator: "\n")
        return MLXLMCommon.Chat.Message(role: role, content: content, images: images)
    }

    // MARK: - Tool Conversion

    private func convertToolToMLXSpec(_ tool: any Tool) -> ToolSpec {
        // Convert AnyLanguageModel's GenerationSchema to JSON-compatible dictionary
        let parametersDict: [String: any Sendable]
        do {
            let resolvedSchema = tool.parameters.withResolvedRoot() ?? tool.parameters
            let encoder = JSONEncoder()
            let data = try encoder.encode(resolvedSchema)
            if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
                parametersDict = try convertToSendableJSONObject(json)
            } else {
                parametersDict = makeEmptyJSONSchemaObject()
            }
        } catch {
            parametersDict = makeEmptyJSONSchemaObject()
        }

        let functionSpec: [String: any Sendable] = [
            "name": tool.name,
            "description": tool.description,
            "parameters": parametersDict,
        ]

        let toolSpec: ToolSpec = [
            "type": "function",
            "function": functionSpec,
        ]

        return toolSpec
    }

    private func makeEmptyJSONSchemaObject() -> [String: any Sendable] {
        [
            "type": "object",
            "properties": [String: any Sendable](),
            "required": [String](),
        ]
    }

    private func convertToSendableJSONObject(_ object: [String: Any]) throws -> [String: any Sendable] {
        var converted: [String: any Sendable] = [:]
        converted.reserveCapacity(object.count)

        for (key, value) in object {
            converted[key] = try convertToSendableJSONValue(value)
        }
        return converted
    }

    private func convertToSendableJSONValue(_ value: Any) throws -> any Sendable {
        if value is NSNull { return MLXLMCommon.JSONValue.null }
        if let stringValue = value as? String { return stringValue }
        if let boolValue = value as? Bool { return boolValue }
        if let intValue = value as? Int { return intValue }
        if let doubleValue = value as? Double { return doubleValue }
        if let numberValue = value as? NSNumber {
            if CFGetTypeID(numberValue) == CFBooleanGetTypeID() {
                return numberValue.boolValue
            }
            return numberValue.doubleValue
        }
        if let arrayValue = value as? [Any] {
            return try arrayValue.map { try convertToSendableJSONValue($0) }
        }
        if let dictionaryValue = value as? [String: Any] {
            return try convertToSendableJSONObject(dictionaryValue)
        }

        throw StructuredGenerationError.invalidTokenization
    }

    // MARK: - Tool Invocation Handling

    private struct ToolInvocationResult {
        let call: Transcript.ToolCall
        let output: Transcript.ToolOutput
    }

    private func resolveToolCalls(
        _ toolCalls: [MLXLMCommon.ToolCall],
        session: LanguageModelSession
    ) async throws -> [ToolInvocationResult] {
        if toolCalls.isEmpty { return [] }

        var toolsByName: [String: any Tool] = [:]
        for tool in session.tools {
            if toolsByName[tool.name] == nil {
                toolsByName[tool.name] = tool
            }
        }

        var results: [ToolInvocationResult] = []
        results.reserveCapacity(toolCalls.count)

        for call in toolCalls {
            let args = try toGeneratedContent(call.function.arguments)
            let callID = UUID().uuidString
            let transcriptCall = Transcript.ToolCall(
                id: callID,
                toolName: call.function.name,
                arguments: args
            )

            guard let tool = toolsByName[call.function.name] else {
                let message = Transcript.Segment.text(.init(content: "Tool not found: \(call.function.name)"))
                let output = Transcript.ToolOutput(
                    id: callID,
                    toolName: call.function.name,
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

    private func toGeneratedContent(_ args: [String: MLXLMCommon.JSONValue]) throws -> GeneratedContent {
        let data = try JSONEncoder().encode(args)
        let json = String(data: data, encoding: .utf8) ?? "{}"
        return try GeneratedContent(json: json)
    }

    private func toolOutputToJSON(_ output: Transcript.ToolOutput) -> String {
        // Extract text content from segments
        var textParts: [String] = []
        for segment in output.segments {
            switch segment {
            case .text(let textSegment):
                textParts.append(textSegment.content)
            case .structure(let structuredSegment):
                // structured content already has jsonString property
                textParts.append(structuredSegment.content.jsonString)
            case .image:
                // Image segments are not supported in MLX tool output
                break
            }
        }
        return textParts.joined(separator: "\n")
    }

    // MARK: - Structured JSON Generation (logit constrained)

    private enum StructuredGenerationError: Error {
        case missingTokenizer
        case emptyPrompt
        case invalidQuoteToken
        case invalidTokenization
        case tokenBudgetExceeded
        case invalidVocabSize
    }

    private func generateStructuredJSON(
        session: LanguageModelSession,
        prompt: Prompt,
        context: ModelContext,
        options: GenerationOptions,
        schema: GenerationSchema
    ) async throws -> String {
        let structuredMaxTokens = options.maximumResponseTokens ?? 512
        let generateParameters = toGenerateParameters(options)

        let chat = convertTranscriptToMLXChat(session: session, fallbackPrompt: prompt.description)
        let userInput = MLXLMCommon.UserInput(
            chat: chat,
            processing: .init(resize: .init(width: 512, height: 512)),
            tools: nil
        )
        let lmInput = try await context.processor.prepare(input: userInput)

        var decoder = try MLXTokenDecoder(
            context: context,
            input: lmInput,
            parameters: generateParameters,
            maximumTokens: structuredMaxTokens
        )

        let vocabSize = decoder.vocabSize
        var generator = try StructuredJSONGenerator(
            schema: schema,
            tokenizeFragment: { fragment in
                context.tokenizer.encode(text: fragment, addSpecialTokens: false)
            },
            tokenText: { token in
                context.tokenizer.decode(tokens: [token], skipSpecialTokens: false)
            },
            decodeToken: { token in
                try decoder.decodeToken(token)
            },
            sampleToken: { allowedTokens in
                try decoder.sampleToken(allowedTokens: allowedTokens)
            },
            maximumTokens: structuredMaxTokens,
            vocabSize: vocabSize
        )

        let json = try generator.generate()
        Stream().synchronize()
        return json
    }

    private struct MLXTokenDecoder {
        let model: any MLXLMCommon.LanguageModel
        var state: MLXLMCommon.LMOutput.State?
        var cache: [MLXLMCommon.KVCache]
        var processor: MLXLMCommon.LogitProcessor?
        let sampler: MLXLMCommon.LogitSampler

        var currentLogits: MLXArray
        let vocabSize: Int

        init(
            context: ModelContext,
            input: MLXLMCommon.LMInput,
            parameters: MLXLMCommon.GenerateParameters,
            maximumTokens: Int
        ) throws {
            self.model = context.model
            self.state = nil
            self.cache = context.model.newCache(parameters: parameters)
            self.processor = parameters.processor()
            self.sampler = parameters.sampler()

            processor?.prompt(input.text.tokens)

            let prepareResult = try context.model.prepare(
                input,
                cache: cache,
                windowSize: parameters.prefillStepSize
            )

            let output: MLXLMCommon.LMOutput
            switch prepareResult {
            case .tokens(let tokensToProcess):
                output = context.model(
                    tokensToProcess[text: .newAxis],
                    cache: cache,
                    state: state
                )
            case .logits(let logitsOutput):
                output = logitsOutput
            }

            self.state = output.state
            self.currentLogits = output.logits

            guard output.logits.shape.count >= 1 else {
                throw StructuredGenerationError.invalidVocabSize
            }
            self.vocabSize = output.logits.shape.last ?? 0
            guard self.vocabSize > 0 else {
                throw StructuredGenerationError.invalidVocabSize
            }
        }

        mutating func decodeToken(_ token: Int) throws {
            let tokenArray = MLXArray(token)
            processor?.didSample(token: tokenArray)

            let inputText = MLXLMCommon.LMInput.Text(tokens: tokenArray)
            let output = model(
                inputText[text: .newAxis],
                cache: cache.isEmpty ? nil : cache,
                state: state
            )
            state = output.state
            currentLogits = output.logits
        }

        mutating func sampleToken(allowedTokens: Set<Int>) throws -> Int {
            guard !allowedTokens.isEmpty else { throw StructuredGenerationError.invalidTokenization }

            var logits = currentLogits[0..., -1, 0...]
            logits = processor?.process(logits: logits) ?? logits
            if logits.dtype == .bfloat16 {
                logits = logits.asType(.float32)
            }

            let allowedIndices = MLXArray(allowedTokens.map { UInt32($0) })
            let maskedLogits = full(logits.shape, values: -Float.infinity)
            maskedLogits[0..., allowedIndices] = logits[0..., allowedIndices]

            let sampledToken = sampler.sample(logits: maskedLogits)
            processor?.didSample(token: sampledToken)
            return sampledToken.item(Int.self)
        }
    }

    private struct StructuredJSONGenerator {
        let schema: GenerationSchema
        let tokenizeFragment: (String) throws -> [Int]
        let tokenText: (Int) -> String
        let decodeToken: (Int) throws -> Void
        let sampleToken: (Set<Int>) throws -> Int

        var remainingTokens: Int
        let totalTokenBudget: Int

        let quoteToken: Int
        let digitOnlyTokens: Set<Int>
        let validStringTokens: Set<Int>
        let validStringTokensOrQuote: Set<Int>

        init(
            schema: GenerationSchema,
            tokenizeFragment: @escaping (String) throws -> [Int],
            tokenText: @escaping (Int) -> String,
            decodeToken: @escaping (Int) throws -> Void,
            sampleToken: @escaping (Set<Int>) throws -> Int,
            maximumTokens: Int,
            vocabSize: Int
        ) throws {
            self.schema = schema
            self.tokenizeFragment = tokenizeFragment
            self.tokenText = tokenText
            self.decodeToken = decodeToken
            self.sampleToken = sampleToken
            self.remainingTokens = maximumTokens
            self.totalTokenBudget = maximumTokens

            let quoteTokens = try tokenizeFragment("\"")
            guard quoteTokens.count == 1, let quoteToken = quoteTokens.first else {
                throw StructuredGenerationError.invalidQuoteToken
            }
            self.quoteToken = quoteToken

            self.digitOnlyTokens = StructuredJSONGenerator.buildDigitOnlyTokens(
                vocabSize: vocabSize,
                tokenText: tokenText
            )
            self.validStringTokens = StructuredJSONGenerator.buildValidJSONStringContentTokens(
                vocabSize: vocabSize,
                tokenText: tokenText
            )
            var tokensOrQuote = self.validStringTokens
            tokensOrQuote.insert(quoteToken)
            self.validStringTokensOrQuote = tokensOrQuote
        }

        mutating func generate() throws -> String {
            try generateNode(schema.root)
        }

        private func maxTokenCountForFreeString() -> Int {
            let perStringLimit = max(32, totalTokenBudget / 4)
            return min(remainingTokens, perStringLimit)
        }

        private static func buildDigitOnlyTokens(
            vocabSize: Int,
            tokenText: (Int) -> String
        ) -> Set<Int> {
            Set((0 ..< vocabSize).filter { tokenId in
                let text = tokenText(tokenId)
                guard !text.isEmpty else { return false }
                return text.allSatisfy({ $0.isNumber })
            })
        }

        private static func buildValidJSONStringContentTokens(
            vocabSize: Int,
            tokenText: (Int) -> String
        ) -> Set<Int> {
            var allowed = Set<Int>()
            allowed.reserveCapacity(vocabSize / 4)

            for tokenId in 0 ..< vocabSize {
                let text = tokenText(tokenId)
                guard !text.isEmpty else { continue }
                if text.contains("\"") { continue }
                if text.contains("\\") { continue }
                if text.unicodeScalars.contains(where: { $0.value < 0x20 }) { continue }
                allowed.insert(tokenId)
            }
            return allowed
        }

        private mutating func emitLiteral(_ text: String) throws -> String {
            for token in try tokenizeFragment(text) {
                guard remainingTokens > 0 else { throw StructuredGenerationError.tokenBudgetExceeded }
                try decodeToken(token)
                remainingTokens -= 1
            }
            return text
        }

        private mutating func generateFreeString(maxTokens: Int) throws -> String {
            var result = ""
            var generatedTokens = 0

            while remainingTokens > 0, generatedTokens < maxTokens {
                let allowedTokens = result.isEmpty ? validStringTokens : validStringTokensOrQuote
                let token = try sampleToken(allowedTokens)
                if token == quoteToken { break }

                result += tokenText(token)
                generatedTokens += 1
                try decodeToken(token)
                remainingTokens -= 1
            }

            return result
        }

        private mutating func generateLiteralChoice(_ candidates: [String]) throws -> String {
            let tokenizedCandidates = try candidates.map { try tokenizeFragment($0) }.filter { !$0.isEmpty }
            guard !tokenizedCandidates.isEmpty else { throw StructuredGenerationError.invalidTokenization }

            var prefixes = tokenizedCandidates
            var emitted = ""
            var tokenPosition = 0

            while remainingTokens > 0 {
                if prefixes.contains(where: { $0.count == tokenPosition }) { break }

                let allowed = Set(prefixes.compactMap { tokens -> Int? in
                    guard tokenPosition < tokens.count else { return nil }
                    return tokens[tokenPosition]
                })

                let nextToken = try sampleToken(allowed)
                emitted += tokenText(nextToken)
                try decodeToken(nextToken)
                remainingTokens -= 1

                prefixes = prefixes.filter { tokens in
                    tokenPosition < tokens.count && tokens[tokenPosition] == nextToken
                }
                tokenPosition += 1
                if prefixes.isEmpty { break }
            }

            return emitted
        }

        private mutating func generateNumber(isInteger: Bool) throws -> String {
            let maxTokens = isInteger ? 3 : 4
            var generated = ""

            for _ in 0 ..< maxTokens {
                guard remainingTokens > 0 else { break }
                let token = try sampleToken(digitOnlyTokens)
                generated += tokenText(token)
                try decodeToken(token)
                remainingTokens -= 1
                if !generated.isEmpty { break }
            }

            return generated.isEmpty ? "0" : generated
        }

        private mutating func generateArray(_ arrayNode: GenerationSchema.ArrayNode) throws -> String {
            let elementCount = arrayNode.minItems ?? arrayNode.maxItems ?? 4
            var output = try emitLiteral("[")

            for index in 0 ..< elementCount {
                output += try generateNode(arrayNode.items)
                if index < elementCount - 1 {
                    output += try emitLiteral(",")
                }
            }

            output += try emitLiteral("]")
            return output
        }

        private mutating func generateObject(_ objectNode: GenerationSchema.ObjectNode) throws -> String {
            let keys = objectNode.properties.keys.sorted()
            var output = try emitLiteral("{")

            for (index, key) in keys.enumerated() {
                output += try emitLiteral("\"")
                output += try emitLiteral(key)
                output += try emitLiteral("\":")

                if let propertyNode = objectNode.properties[key] {
                    output += try generateNode(propertyNode)
                } else {
                    output += try emitLiteral("null")
                }

                if index < keys.count - 1 {
                    output += try emitLiteral(",")
                }
            }

            output += try emitLiteral("}")
            return output
        }

        private mutating func generateNode(_ node: GenerationSchema.Node) throws -> String {
            guard remainingTokens > 0 else { throw StructuredGenerationError.tokenBudgetExceeded }

            switch node {
            case .string(let stringNode):
                var output = try emitLiteral("\"")
                if let enumChoices = stringNode.enumChoices, !enumChoices.isEmpty {
                    output += try generateLiteralChoice(enumChoices)
                } else {
                    output += try generateFreeString(maxTokens: maxTokenCountForFreeString())
                }
                output += try emitLiteral("\"")
                return output

            case .number(let numberNode):
                return try generateNumber(isInteger: numberNode.integerOnly)

            case .boolean:
                return try generateLiteralChoice(["true", "false"])

            case .array(let arrayNode):
                return try generateArray(arrayNode)

            case .object(let objectNode):
                return try generateObject(objectNode)

            case .anyOf(let nodes):
                guard let first = nodes.first else { throw StructuredGenerationError.invalidTokenization }
                return try generateNode(first)

            case .ref(let refName):
                guard let referenced = schema.defs[refName] else { throw StructuredGenerationError.invalidTokenization }
                return try generateNode(referenced)
            }
        }
    }
#endif  // MLX
