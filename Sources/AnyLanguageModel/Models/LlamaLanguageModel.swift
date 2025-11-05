import Foundation
#if Llama
    import Llama

    /// A language model that runs llama.cpp models locally.
    ///
    /// Use this model to generate text using GGUF models running directly with llama.cpp.
    ///
    /// ```swift
    /// let model = LlamaLanguageModel(
    ///     modelPath: "/path/to/model.gguf",
    ///     contextSize: 2048
    /// )
    /// ```
    public final class LlamaLanguageModel: LanguageModel, @unchecked Sendable {
        /// The reason the model is unavailable.
        /// This model is always available.
        public typealias UnavailableReason = Never

        /// The path to the GGUF model file.
        public let modelPath: String

        /// The context size for the model.
        public let contextSize: UInt32

        /// The batch size for processing.
        public let batchSize: UInt32

        /// The number of threads to use.
        public let threads: Int32

        /// The random seed for generation.
        public let seed: UInt32

        /// The temperature for sampling.
        public let temperature: Float

        /// The top-K sampling parameter.
        public let topK: Int32

        /// The top-P (nucleus) sampling parameter.
        public let topP: Float

        /// The repeat penalty for generation.
        public let repeatPenalty: Float

        /// The number of tokens to consider for repeat penalty.
        public let repeatLastN: Int32

        /// The loaded model instance
        private var model: OpaquePointer?

        /// The model's vocabulary
        private var vocab: OpaquePointer?

        /// Whether the model is currently loaded
        private var isModelLoaded: Bool = false

        /// Creates a Llama language model.
        ///
        /// - Parameters:
        ///   - modelPath: The path to the GGUF model file.
        ///   - contextSize: The context size for the model. Defaults to 2048.
        ///   - batchSize: The batch size for processing. Defaults to 512.
        ///   - threads: The number of threads to use. Defaults to the number of processors.
        ///   - seed: The random seed for generation. Defaults to a random value.
        ///   - temperature: The temperature for sampling. Defaults to 0.8.
        ///   - topK: The top-K sampling parameter. Defaults to 40.
        ///   - topP: The top-P (nucleus) sampling parameter. Defaults to 0.95.
        ///   - repeatPenalty: The repeat penalty for generation. Defaults to 1.1.
        ///   - repeatLastN: The number of tokens to consider for repeat penalty. Defaults to 64.
        public init(
            modelPath: String,
            contextSize: UInt32 = 2048,
            batchSize: UInt32 = 512,
            threads: Int32 = Int32(ProcessInfo.processInfo.processorCount),
            seed: UInt32 = UInt32.random(in: 0 ... UInt32.max),
            temperature: Float = 0.8,
            topK: Int32 = 40,
            topP: Float = 0.95,
            repeatPenalty: Float = 1.1,
            repeatLastN: Int32 = 64
        ) {
            self.modelPath = modelPath
            self.contextSize = contextSize
            self.batchSize = batchSize
            self.threads = threads
            self.seed = seed
            self.temperature = temperature
            self.topK = topK
            self.topP = topP
            self.repeatPenalty = repeatPenalty
            self.repeatLastN = repeatLastN
        }

        deinit {
            if let model = model {
                llama_model_free(model)
            }
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
                fatalError("LlamaLanguageModel only supports generating String content")
            }

            try await ensureModelLoaded()

            let contextParams = createContextParams(from: options)

            // Try to create context with error handling
            guard let context = llama_init_from_model(model!, contextParams) else {
                throw LlamaLanguageModelError.contextInitializationFailed
            }

            defer { llama_free(context) }

            llama_set_causal_attn(context, true)
            llama_set_warmup(context, false)
            llama_set_n_threads(context, threads, threads)

            let maxTokens = options.maximumResponseTokens ?? 100
            let text = try await generateText(
                context: context,
                model: model!,
                prompt: prompt.description,
                maxTokens: maxTokens,
                options: options
            )

            return LanguageModelSession.Response(
                content: text as! Content,
                rawContent: GeneratedContent(text),
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
                fatalError("LlamaLanguageModel only supports generating String content")
            }

            let maxTokens = options.maximumResponseTokens ?? 100

            let stream: AsyncThrowingStream<LanguageModelSession.ResponseStream<Content>.Snapshot, any Error> =
                AsyncThrowingStream { continuation in
                    let task = Task {
                        do {
                            try await ensureModelLoaded()

                            let contextParams = createContextParams(from: options)
                            guard let context = llama_init_from_model(model!, contextParams) else {
                                throw LlamaLanguageModelError.contextInitializationFailed
                            }
                            defer { llama_free(context) }

                            // Stabilize runtime behavior per-context
                            llama_set_causal_attn(context, true)
                            llama_set_warmup(context, false)
                            llama_set_n_threads(context, self.threads, self.threads)

                            var accumulatedText = ""

                            do {
                                for try await tokenText in generateTextStream(
                                    context: context,
                                    model: model!,
                                    prompt: prompt.description,
                                    maxTokens: maxTokens,
                                    options: options
                                ) {
                                    accumulatedText += tokenText

                                    let snapshot = LanguageModelSession.ResponseStream<Content>.Snapshot(
                                        content: (accumulatedText as! Content).asPartiallyGenerated(),
                                        rawContent: GeneratedContent(accumulatedText)
                                    )
                                    continuation.yield(snapshot)
                                }
                            } catch {
                                continuation.finish(throwing: error)
                                return
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

        // MARK: - Private Helpers

        private func ensureModelLoaded() async throws {
            guard !isModelLoaded else { return }

            // Check if model file exists
            guard FileManager.default.fileExists(atPath: modelPath) else {
                throw LlamaLanguageModelError.invalidModelPath
            }

            // Initialize backend lazily - must be done before loading model
            llama_backend_init()

            // Free any existing model before loading a new one
            if let existingModel = model {
                llama_model_free(existingModel)
                self.model = nil
            }

            let modelParams = createModelParams()
            guard let loadedModel = llama_model_load_from_file(modelPath, modelParams) else {
                throw LlamaLanguageModelError.modelLoadFailed
            }

            self.model = loadedModel
            self.vocab = llama_model_get_vocab(loadedModel)
            self.isModelLoaded = true
        }

        private func createModelParams() -> llama_model_params {
            var params = llama_model_default_params()

            // Force CPU-only execution to avoid Metal GPU issues
            params.n_gpu_layers = 0

            // Try to reduce memory usage
            params.use_mmap = true
            params.use_mlock = false
            return params
        }

        private func createContextParams(from options: GenerationOptions) -> llama_context_params {
            var params = llama_context_default_params()
            params.n_ctx = contextSize
            params.n_batch = batchSize
            params.n_threads = threads
            params.n_threads_batch = threads
            return params
        }

        private func generateText(
            context: OpaquePointer,
            model: OpaquePointer,
            prompt: String,
            maxTokens: Int,
            options: GenerationOptions
        ) async throws
            -> String
        {
            guard let vocab = llama_model_get_vocab(model) else {
                throw LlamaLanguageModelError.contextInitializationFailed
            }

            // Tokenize the prompt
            let promptTokens = try tokenizeText(vocab: vocab, text: prompt)
            guard !promptTokens.isEmpty else {
                throw LlamaLanguageModelError.tokenizationFailed
            }

            var batch = llama_batch_init(Int32(batchSize), 0, 1)
            defer { llama_batch_free(batch) }

            batch.n_tokens = Int32(promptTokens.count)
            for i in 0 ..< promptTokens.count {
                let idx = Int(i)
                batch.token[idx] = promptTokens[idx]
                batch.pos[idx] = Int32(i)
                batch.n_seq_id[idx] = 1
                if let seq_ids = batch.seq_id, let seq_id = seq_ids[idx] {
                    seq_id[0] = 0
                }
                batch.logits[idx] = 0
            }

            if batch.n_tokens > 0 {
                batch.logits[Int(batch.n_tokens) - 1] = 1
            }

            guard llama_decode(context, batch) == 0 else {
                throw LlamaLanguageModelError.encodingFailed
            }

            // Initialize sampler chain with options
            guard let sampler = llama_sampler_chain_init(llama_sampler_chain_default_params()) else {
                throw LlamaLanguageModelError.decodingFailed
            }
            defer { llama_sampler_free(sampler) }

            // Use sampling parameters from options if provided
            if let sampling = options.sampling {
                switch sampling.mode {
                case .greedy:
                    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(1))
                    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(1.0, 1))
                case .topK(let k, let seed):
                    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(Int32(k)))
                    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(1.0, 1))
                    if let temperature = options.temperature {
                        llama_sampler_chain_add(sampler, llama_sampler_init_temp(Float(temperature)))
                    }
                    if let seed = seed {
                        llama_sampler_chain_add(sampler, llama_sampler_init_dist(UInt32(seed)))
                    }
                case .nucleus(let threshold, let seed):
                    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(0))  // Disable top-k
                    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(Float(threshold), 1))
                    if let temperature = options.temperature {
                        llama_sampler_chain_add(sampler, llama_sampler_init_temp(Float(temperature)))
                    }
                    if let seed = seed {
                        llama_sampler_chain_add(sampler, llama_sampler_init_dist(UInt32(seed)))
                    }
                }
            } else {
                // Use model's default sampling parameters
                if topK > 0 { llama_sampler_chain_add(sampler, llama_sampler_init_top_k(topK)) }
                if topP < 1.0 { llama_sampler_chain_add(sampler, llama_sampler_init_top_p(topP, 1)) }
                llama_sampler_chain_add(sampler, llama_sampler_init_dist(seed))
            }

            // Generate tokens one by one
            var generatedText = ""
            var n_cur = batch.n_tokens

            for _ in 0 ..< maxTokens {
                // Sample next token from logits - llama_batch_get_one creates batch with single token at index 0
                let nextToken = llama_sampler_sample(sampler, context, batch.n_tokens - 1)
                llama_sampler_accept(sampler, nextToken)

                // Check for end of sequence
                if llama_vocab_is_eog(vocab, nextToken) {
                    break
                }

                // Convert token to text
                if let tokenText = tokenToText(vocab: vocab, token: nextToken) {
                    generatedText += tokenText
                }

                // Prepare batch for next token
                batch.n_tokens = 1
                batch.token[0] = nextToken
                batch.pos[0] = n_cur
                batch.n_seq_id[0] = 1
                if let seq_ids = batch.seq_id, let seq_id = seq_ids[0] {
                    seq_id[0] = 0
                }
                batch.logits[0] = 1

                n_cur += 1

                let decodeResult = llama_decode(context, batch)
                guard decodeResult == 0 else {
                    break
                }
            }

            return generatedText
        }

        private func generateTextStream(
            context: OpaquePointer,
            model: OpaquePointer,
            prompt: String,
            maxTokens: Int,
            options: GenerationOptions
        )
            -> AsyncThrowingStream<String, Error>
        {
            return AsyncThrowingStream { continuation in
                self.performTextGeneration(
                    context: context,
                    model: model,
                    prompt: prompt,
                    maxTokens: maxTokens,
                    options: options,
                    continuation: continuation
                )
            }
        }

        private func performTextGeneration(
            context: OpaquePointer,
            model: OpaquePointer,
            prompt: String,
            maxTokens: Int,
            options: GenerationOptions,
            continuation: AsyncThrowingStream<String, Error>.Continuation
        ) {
            do {
                guard let vocab = llama_model_get_vocab(model) else {
                    continuation.finish(throwing: LlamaLanguageModelError.contextInitializationFailed)
                    return
                }

                // Tokenize the prompt
                let promptTokens = try tokenizeText(vocab: vocab, text: prompt)
                guard !promptTokens.isEmpty else {
                    continuation.finish(throwing: LlamaLanguageModelError.tokenizationFailed)
                    return
                }

                // Initialize batch
                var batch = llama_batch_init(Int32(batchSize), 0, 1)
                defer { llama_batch_free(batch) }

                // Evaluate the prompt
                batch.n_tokens = Int32(promptTokens.count)
                for i in 0 ..< promptTokens.count {
                    let idx = Int(i)
                    batch.token[idx] = promptTokens[idx]
                    batch.pos[idx] = Int32(i)
                    batch.n_seq_id[idx] = 1
                    if let seq_ids = batch.seq_id, let seq_id = seq_ids[idx] {
                        seq_id[0] = 0
                    }
                    batch.logits[idx] = 0
                }
                if batch.n_tokens > 0 {
                    batch.logits[Int(batch.n_tokens) - 1] = 1
                }

                guard llama_decode(context, batch) == 0 else {
                    throw LlamaLanguageModelError.encodingFailed
                }

                // Initialize sampler chain with options
                guard let sampler = llama_sampler_chain_init(llama_sampler_chain_default_params()) else {
                    throw LlamaLanguageModelError.decodingFailed
                }
                defer { llama_sampler_free(sampler) }

                // Use sampling parameters from options if provided
                if let sampling = options.sampling {
                    switch sampling.mode {
                    case .greedy:
                        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(1))
                        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(1.0, 1))
                    case .topK(let k, let seed):
                        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(Int32(k)))
                        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(1.0, 1))
                        if let temperature = options.temperature {
                            llama_sampler_chain_add(sampler, llama_sampler_init_temp(Float(temperature)))
                        }
                        if let seed = seed {
                            llama_sampler_chain_add(sampler, llama_sampler_init_dist(UInt32(seed)))
                        }
                    case .nucleus(let threshold, let seed):
                        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(0))  // Disable top-k
                        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(Float(threshold), 1))
                        if let temperature = options.temperature {
                            llama_sampler_chain_add(sampler, llama_sampler_init_temp(Float(temperature)))
                        }
                        if let seed = seed {
                            llama_sampler_chain_add(sampler, llama_sampler_init_dist(UInt32(seed)))
                        }
                    }
                } else {
                    // Use model's default sampling parameters
                    if self.topK > 0 { llama_sampler_chain_add(sampler, llama_sampler_init_top_k(self.topK)) }
                    if self.topP < 1.0 { llama_sampler_chain_add(sampler, llama_sampler_init_top_p(self.topP, 1)) }
                    llama_sampler_chain_add(sampler, llama_sampler_init_dist(self.seed))
                }

                // Generate tokens one by one
                var n_cur = batch.n_tokens

                for _ in 0 ..< maxTokens {
                    // Sample next token from logits of the last token we just decoded
                    let nextToken = llama_sampler_sample(sampler, context, batch.n_tokens - 1)
                    llama_sampler_accept(sampler, nextToken)

                    // Check for end of sequence
                    if llama_vocab_is_eog(vocab, nextToken) {
                        break
                    }

                    // Convert token to text and yield it
                    if let tokenText = tokenToText(vocab: vocab, token: nextToken) {
                        continuation.yield(tokenText)
                    }

                    // Prepare batch for next token
                    batch.n_tokens = 1
                    batch.token[0] = nextToken
                    batch.pos[0] = n_cur
                    batch.n_seq_id[0] = 1
                    if let seq_ids = batch.seq_id, let seq_id = seq_ids[0] {
                        seq_id[0] = 0
                    }
                    batch.logits[0] = 1

                    n_cur += 1

                    let decodeResult = llama_decode(context, batch)
                    guard decodeResult == 0 else {
                        break
                    }
                }

                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }

        // MARK: - Helper Methods

        private func tokenizeText(vocab: OpaquePointer, text: String) throws -> [llama_token] {
            let utf8Count = text.utf8.count
            let maxTokens = Int32(max(utf8Count * 2, 8))  // Rough estimate, minimum capacity
            let tokens = UnsafeMutablePointer<llama_token>.allocate(capacity: Int(maxTokens))
            defer { tokens.deallocate() }

            let tokenCount = llama_tokenize(
                vocab,
                text,
                Int32(utf8Count),
                tokens,
                maxTokens,
                true,  // addSpecial
                true  // parseSpecial
            )

            guard tokenCount > 0 else {
                throw LlamaLanguageModelError.tokenizationFailed
            }

            return Array(UnsafeBufferPointer(start: tokens, count: Int(tokenCount)))
        }

        private func tokenToText(vocab: OpaquePointer, token: llama_token) -> String? {
            // First attempt with a reasonable buffer
            var cap: Int32 = 64
            var buf = UnsafeMutablePointer<CChar>.allocate(capacity: Int(cap))
            defer { buf.deallocate() }

            var written = llama_token_to_piece(
                vocab,
                token,
                buf,
                cap,
                0,
                false
            )

            if written < 0 {
                // Reallocate to the required size and retry
                cap = -written
                buf.deallocate()
                buf = UnsafeMutablePointer<CChar>.allocate(capacity: Int(cap))
                written = llama_token_to_piece(
                    vocab,
                    token,
                    buf,
                    cap,
                    0,
                    false
                )
            }

            let count = Int(max(0, written))
            if count == 0 { return nil }

            // Create String from exact byte count (no reliance on NUL termination)
            let rawPtr = UnsafeRawPointer(buf)
            let u8Ptr = rawPtr.assumingMemoryBound(to: UInt8.self)
            let bytes = UnsafeBufferPointer(start: u8Ptr, count: count)
            return String(decoding: bytes, as: UTF8.self)
        }
    }

    /// Errors that can occur when using LlamaLanguageModel
    public enum LlamaLanguageModelError: Error, LocalizedError {
        case modelLoadFailed
        case contextInitializationFailed
        case tokenizationFailed
        case encodingFailed
        case decodingFailed
        case invalidModelPath
        case insufficientMemory
        case unsupportedFeature

        public var errorDescription: String? {
            switch self {
            case .modelLoadFailed:
                return "Failed to load model from file"
            case .contextInitializationFailed:
                return "Failed to initialize context"
            case .tokenizationFailed:
                return "Failed to tokenize input text"
            case .encodingFailed:
                return "Failed to encode prompt"
            case .decodingFailed:
                return "Failed to decode response"
            case .invalidModelPath:
                return "Invalid model file path"
            case .insufficientMemory:
                return "Insufficient memory for operation"
            case .unsupportedFeature:
                return "This LlamaLanguageModel does not support image segments"
            }
        }
    }
#endif  // Llama
