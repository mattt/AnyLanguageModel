import Foundation
import CoreML
import Tokenizers
@preconcurrency import Generation
@preconcurrency import Models

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
public struct CoreMLLanguageModel: AnyLanguageModel.LanguageModel {
    private let model: Models.LanguageModel
    private let tokenizer: Tokenizer

    public init(url: URL, computeUnits: MLComputeUnits = .all) async throws {
        // Load the CoreML model from the specified path and compile it, if needed
        let compiledURL = try compileModel(at: url)

        // Load the model with the specified compute units
        self.model = try Models.LanguageModel.loadCompiled(url: compiledURL, computeUnits: computeUnits)

        // Load the tokenizer
        self.tokenizer = try await model.tokenizer
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

        // Build chat with optional system instructions
        var chat: [Message] = []
        if let instructions = session.instructions?.description, !instructions.isEmpty {
            chat.append(["role": "system", "content": instructions])
        }
        chat.append(["role": "user", "content": prompt.description])

        // Convert tools to Tokenizers.ToolSpec when available
        let toolSpecs: [ToolSpec]? =
            session.tools.isEmpty
            ? nil
            : session.tools.map { tool in toToolSpec(tool) }

        let tokens = try tokenizer.applyChatTemplate(messages: chat, tools: toolSpecs)

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
                    // Build chat with optional system instructions
                    var chat: [Message] = []
                    if let instructions = session.instructions?.description, !instructions.isEmpty {
                        chat.append(["role": "system", "content": instructions])
                    }
                    chat.append(["role": "user", "content": prompt.description])

                    // Convert tools to Tokenizers.ToolSpec when available
                    let toolSpecs: [ToolSpec]? =
                        session.tools.isEmpty
                        ? nil
                        : session.tools.map { tool in toToolSpec(tool) }

                    let tokens = try tokenizer.applyChatTemplate(messages: chat, tools: toolSpecs)

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

// Convert AnyLanguageModel Tool into a Tokenizers.ToolSpec dictionary understood by chat templates
private func toToolSpec(_ tool: any Tool) -> ToolSpec {
    // Convert AnyLanguageModel's GenerationSchema to JSON-compatible dictionary
    let parametersDict: [String: Any]
    do {
        let encoder = JSONEncoder()
        let data = try encoder.encode(tool.parameters)
        if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
            // Resolve $ref if present
            if let ref = json["$ref"] as? String,
                let defs = json["$defs"] as? [String: Any]
            {
                let defName = ref.replacingOccurrences(of: "#/$defs/", with: "")
                if let resolvedDef = defs[defName] as? [String: Any] {
                    parametersDict = resolvedDef
                } else {
                    parametersDict = ["type": "object", "properties": [:], "required": []]
                }
            } else {
                parametersDict = json
            }
        } else {
            parametersDict = ["type": "object", "properties": [:], "required": []]
        }
    } catch {
        parametersDict = ["type": "object", "properties": [:], "required": []]
    }

    return [
        "type": "function",
        "function": [
            "name": tool.name,
            "description": tool.description,
            "parameters": parametersDict,
        ],
    ]
}

private func compileModel(at url: URL) throws -> URL {
    #if os(watchOS)
        fatalError("Model compilation is not supported on watchOS")
    #else
        if url.pathExtension == "mlmodelc" {
            return url
        }

        let modelName = url.deletingPathExtension().lastPathComponent
        let cacheBase = try FileManager.default.url(
            for: .cachesDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        let cacheRoot = cacheBase.appendingPathComponent("co.huggingface.AnyLanguageModel", isDirectory: true)
        let cached = cacheRoot.appendingPathComponent("\(modelName).mlmodelc", isDirectory: true)

        if FileManager.default.fileExists(atPath: cached.path) {
            return cached
        }

        print("Compiling model \(url)")
        let compiled = try MLModel.compileModel(at: url)

        try FileManager.default.createDirectory(at: cacheRoot, withIntermediateDirectories: true)
        try? FileManager.default.removeItem(at: cached)
        try FileManager.default.copyItem(at: compiled, to: cached)
        try? FileManager.default.removeItem(at: compiled)
        return cached
    #endif
}
