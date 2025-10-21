import Foundation
import CoreML
import Tokenizers
@preconcurrency import Generation
@preconcurrency import Models

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
public struct CoreMLLanguageModel: AnyLanguageModel.LanguageModel {
    private let model: Models.LanguageModel
    private let tokenizer: Tokenizer
    private let chatTemplateHandler: (@Sendable (Instructions?, Prompt) -> [Message])?
    private let toolsHandler: (@Sendable ([any Tool]) -> [ToolSpec])?

    public init(
        url: URL,
        computeUnits: MLComputeUnits = .all,
        chatTemplateHandler: (@Sendable (Instructions?, Prompt) -> [Message])? = nil,
        toolsHandler: (@Sendable ([any Tool]) -> [ToolSpec])? = nil
    ) async throws {
        // Ensure the model is already compiled
        guard url.pathExtension == "mlmodelc" else {
            throw CoreMLLanguageModelError.compiledModelRequired
        }

        // Load the model with the specified compute units
        self.model = try Models.LanguageModel.loadCompiled(url: url, computeUnits: computeUnits)

        // Load the tokenizer
        self.tokenizer = try await model.tokenizer

        self.chatTemplateHandler = chatTemplateHandler
        self.toolsHandler = toolsHandler
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
            fatalError("CoreMLLanguageModel only supports generating String content")
        }

        // Convert AnyLanguageModel GenerationOptions to swift-transformers GenerationConfig
        let generationConfig = toGenerationConfig(options)

        let tokens: [Int]
        if let chatTemplateHandler = chatTemplateHandler {
            // Use chat template handler with optional tools
            let chat = chatTemplateHandler(session.instructions, prompt)
            let toolSpecs: [ToolSpec]? = toolsHandler?(session.tools)
            tokens = try tokenizer.applyChatTemplate(messages: chat, tools: toolSpecs)
        } else {
            // Fall back to direct tokenizer encoding
            tokens = tokenizer.encode(text: prompt.description)
        }

        // Reset model state for new generation
        await model.resetState()

        let response = await model.generate(
            config: generationConfig,
            tokens: tokens,
            model: model.callAsFunction
        )

        return LanguageModelSession.Response(
            content: response as! Content,
            rawContent: GeneratedContent(response),
            transcriptEntries: ArraySlice([])
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
            fatalError("CoreMLLanguageModel only supports generating String content")
        }

        // Convert AnyLanguageModel GenerationOptions to swift-transformers GenerationConfig
        let generationConfig = toGenerationConfig(options)

        // Transform the generation into ResponseStream snapshots
        let stream: AsyncThrowingStream<LanguageModelSession.ResponseStream<Content>.Snapshot, any Error> = .init {
            @Sendable continuation in
            let task = Task {
                do {
                    let tokens: [Int]
                    if let chatTemplateHandler = chatTemplateHandler {
                        // Use chat template handler with optional tools
                        let chat = chatTemplateHandler(session.instructions, prompt)
                        let toolSpecs: [ToolSpec]? = toolsHandler?(session.tools)
                        tokens = try tokenizer.applyChatTemplate(messages: chat, tools: toolSpecs)
                    } else {
                        // Fall back to direct tokenizer encoding
                        tokens = tokenizer.encode(text: prompt.description)
                    }

                    await model.resetState()

                    // Decode the rendered prompt once to strip it from streamed output
                    let promptTextPrefix = tokenizer.decode(tokens: tokens)
                    var accumulatedText = ""

                    _ = await model.generate(
                        config: generationConfig,
                        tokens: tokens,
                        model: model.callAsFunction
                    ) { tokenIds in
                        // Decode full text and strip the prompt prefix
                        let fullText = tokenizer.decode(tokens: tokenIds)
                        let assistantText: String
                        if fullText.hasPrefix(promptTextPrefix) {
                            let startIdx = fullText.index(fullText.startIndex, offsetBy: promptTextPrefix.count)
                            assistantText = String(fullText[startIdx...])
                        } else {
                            assistantText = fullText
                        }

                        // Compute delta vs accumulated text and yield
                        if assistantText.count >= accumulatedText.count,
                            assistantText.hasPrefix(accumulatedText)
                        {
                            let startIdx = assistantText.index(
                                assistantText.startIndex,
                                offsetBy: accumulatedText.count
                            )
                            let delta = String(assistantText[startIdx...])
                            accumulatedText += delta
                        } else {
                            accumulatedText = assistantText
                        }

                        continuation.yield(
                            .init(
                                content: (accumulatedText as! Content).asPartiallyGenerated(),
                                rawContent: GeneratedContent(accumulatedText)
                            )
                        )
                    }

                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            continuation.onTermination = { _ in
                task.cancel()
            }
        }

        return LanguageModelSession.ResponseStream(stream: stream)
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
public enum CoreMLLanguageModelError: LocalizedError {
    case compiledModelRequired

    public var errorDescription: String? {
        switch self {
        case .compiledModelRequired:
            return
                "A compiled Core ML model (.mlmodelc) is required. Please compile your model first using MLModel.compileModel(at:)."
        }
    }
}

// MARK: -

private func toGenerationConfig(_ options: GenerationOptions) -> GenerationConfig {
    var config = GenerationConfig(maxNewTokens: options.maximumResponseTokens ?? 2048)

    // Map temperature
    if let temperature = options.temperature {
        config.temperature = Float(temperature)
    }

    // Map sampling mode
    if let sampling = options.sampling {
        switch sampling.mode {
        case .greedy:
            config.doSample = false
        case .topK(let k, _):
            config.doSample = true
            config.topK = k
        case .nucleus(let p, _):
            config.doSample = true
            config.topP = p
        }
    }

    return config
}
