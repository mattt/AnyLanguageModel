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

        // User prompt
        let chat: [Message] = [["role" : "user", "content" : prompt.description]]
        let inputIds = try await tokenizer.applyChatTemplate(messages: chat)

        // TODO: we need either one of these (or possibly both):
        // - A version of applyChatTemplate that does not tokenize
        // - A version of Core ML `model.generate` that accepts input ids
        let lmInput = try await tokenizer.decode(tokens: inputIds, skipSpecialTokens: false)

        // Reset model state for new generation
        await model.resetState()

        // Generate response using swift-transformers
        let response = try await model.generate(
            config: generationConfig,
            prompt: lmInput
        ) { _ in
            // No streaming callback needed for non-streaming response
        }

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

        // For streaming, we'll collect the response and return it as a single chunk
        // This is a simplified implementation - a full streaming implementation would
        // require more complex async handling
        _ = Task {
            await model.resetState()
            return try await model.generate(
                config: generationConfig,
                prompt: prompt.description
            ) { _ in
                // No streaming callback needed for this implementation
            }
        }

        // Return a stream that yields the complete response
        return LanguageModelSession.ResponseStream(
            content: "" as! Content,
            rawContent: GeneratedContent("")
        )
    }
}

// MARK: - Options Mapping

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

// MARK: - Model Compilation

private func compileModel(at url: URL) throws -> URL {
    #if os(watchOS)
        fatalError("Model compilation is not supported on watchOS")
    #else
        if url.pathExtension == "mlmodelc" {
            return url
        }

        let modelName = url.deletingPathExtension().lastPathComponent
        let cacheBase = try FileManager.default.url(for: .cachesDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
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
