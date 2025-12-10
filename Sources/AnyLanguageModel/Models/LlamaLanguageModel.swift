import Foundation
#if Llama
    import Llama

    /// Global storage for the current log level threshold.
    /// This is needed because the C callback can't capture Swift context.
    /// Access is synchronized by llama.cpp's internal logging mechanism.
    nonisolated(unsafe) private var currentLogLevel: LlamaLanguageModel.LogLevel = .warn

    /// Custom log callback that filters messages based on the current log level.
    private func llamaLogCallback(
        level: ggml_log_level,
        text: UnsafePointer<CChar>?,
        userData: UnsafeMutableRawPointer?
    ) {
        guard level.rawValue >= currentLogLevel.ggmlLevel.rawValue else { return }
        if let text = text {
            fputs(String(cString: text), stderr)
        }
    }

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

        /// The verbosity level for llama.cpp logging.
        public enum LogLevel: Int, Hashable, Comparable, Sendable, CaseIterable {
            /// No logging output.
            case none = 0
            /// Debug messages and above (most verbose).
            case debug = 1
            /// Info messages and above.
            case info = 2
            /// Warning messages and above (default).
            case warn = 3
            /// Only error messages.
            case error = 4

            /// Maps to the corresponding ggml log level.
            var ggmlLevel: ggml_log_level {
                switch self {
                case .none: return GGML_LOG_LEVEL_NONE
                case .debug: return GGML_LOG_LEVEL_DEBUG
                case .info: return GGML_LOG_LEVEL_INFO
                case .warn: return GGML_LOG_LEVEL_WARN
                case .error: return GGML_LOG_LEVEL_ERROR
                }
            }

            public static func < (lhs: LogLevel, rhs: LogLevel) -> Bool {
                lhs.rawValue < rhs.rawValue
            }
        }

        /// Custom generation options specific to llama.cpp.
        ///
        /// Use this type to pass llama.cpp-specific sampling parameters that are
        /// not part of the standard ``GenerationOptions``.
        ///
        /// ```swift
        /// var options = GenerationOptions(temperature: 0.8)
        /// options[custom: LlamaLanguageModel.self] = .init(
        ///     repeatPenalty: 1.2,
        ///     repeatLastN: 128,
        ///     frequencyPenalty: 0.1,
        ///     presencePenalty: 0.1,
        ///     mirostat: .v2(tau: 5.0, eta: 0.1)
        /// )
        /// ```
        public struct CustomGenerationOptions: AnyLanguageModel.CustomGenerationOptions, Codable {
            /// Context size to allocate for the model.
            public var contextSize: UInt32?

            /// Batch size to use when evaluating tokens.
            public var batchSize: UInt32?

            /// Number of threads to use for computation.
            public var threads: Int32?

            /// Random seed for deterministic sampling.
            public var seed: UInt32?

            /// Sampling temperature.
            public var temperature: Float?

            /// Top-K sampling parameter.
            public var topK: Int32?

            /// Top-P (nucleus) sampling parameter.
            public var topP: Float?

            /// The penalty applied to repeated tokens.
            ///
            /// Values greater than 1.0 discourage repetition, while values less than 1.0
            /// encourage it. A value of 1.0 applies no penalty.
            public var repeatPenalty: Float?

            /// The number of recent tokens to consider for the repeat penalty.
            ///
            /// Only the last `repeatLastN` tokens will be checked for repetition.
            /// Set to 0 to disable repeat penalty, or -1 to consider all tokens.
            public var repeatLastN: Int32?

            /// The frequency penalty applied during sampling.
            ///
            /// Positive values penalize tokens based on their frequency in the text so far,
            /// decreasing the likelihood of repeating the same content.
            public var frequencyPenalty: Float?

            /// The presence penalty applied during sampling.
            ///
            /// Positive values penalize tokens that have appeared at all in the text so far,
            /// encouraging the model to generate novel content.
            public var presencePenalty: Float?

            /// Mirostat sampling configuration for adaptive perplexity control.
            public enum MirostatMode: Hashable, Codable, Sendable {
                /// Mirostat v1 with target entropy (tau) and learning rate (eta).
                case v1(tau: Float, eta: Float)

                /// Mirostat v2 with target entropy (tau) and learning rate (eta).
                case v2(tau: Float, eta: Float)
            }

            /// Mirostat sampling mode for adaptive perplexity control.
            public var mirostat: MirostatMode?

            /// Creates custom generation options for llama.cpp.
            public init(
                contextSize: UInt32? = nil,
                batchSize: UInt32? = nil,
                threads: Int32? = nil,
                seed: UInt32? = nil,
                temperature: Float? = nil,
                topK: Int32? = nil,
                topP: Float? = nil,
                repeatPenalty: Float? = nil,
                repeatLastN: Int32? = nil,
                frequencyPenalty: Float? = nil,
                presencePenalty: Float? = nil,
                mirostat: MirostatMode? = nil
            ) {
                self.contextSize = contextSize
                self.batchSize = batchSize
                self.threads = threads
                self.seed = seed
                self.temperature = temperature
                self.topK = topK
                self.topP = topP
                self.repeatPenalty = repeatPenalty
                self.repeatLastN = repeatLastN
                self.frequencyPenalty = frequencyPenalty
                self.presencePenalty = presencePenalty
                self.mirostat = mirostat
            }

            /// Default llama.cpp options used when none are provided at runtime.
            public static var `default`: Self {
                .init(
                    contextSize: 2048,
                    batchSize: 512,
                    threads: Int32(ProcessInfo.processInfo.processorCount),
                    seed: UInt32.random(in: 0 ... UInt32.max),
                    temperature: 0.8,
                    topK: 40,
                    topP: 0.95,
                    repeatPenalty: 1.1,
                    repeatLastN: 64,
                    frequencyPenalty: 0.0,
                    presencePenalty: 0.0,
                    mirostat: nil
                )
            }

        }

        /// The path to the GGUF model file.
        public let modelPath: String

        /// The context size for the model.
        ///
        /// - Important: This property is deprecated. Use ``GenerationOptions`` with
        ///   custom options instead:
        ///   ```swift
        ///   var options = GenerationOptions()
        ///   options[custom: LlamaLanguageModel.self] = .init(contextSize: 4096)
        ///   ```
        @available(*, deprecated, message: "Use GenerationOptions custom options instead")
        public var contextSize: UInt32 { legacyDefaults.contextSize }

        /// The batch size for processing.
        ///
        /// - Important: This property is deprecated. Use ``GenerationOptions`` with
        ///   custom options instead.
        @available(*, deprecated, message: "Use GenerationOptions custom options instead")
        public var batchSize: UInt32 { legacyDefaults.batchSize }

        /// The number of threads to use.
        ///
        /// - Important: This property is deprecated. Use ``GenerationOptions`` with
        ///   custom options instead.
        @available(*, deprecated, message: "Use GenerationOptions custom options instead")
        public var threads: Int32 { legacyDefaults.threads }

        /// The random seed for generation.
        ///
        /// - Important: This property is deprecated. Use ``GenerationOptions`` with
        ///   custom options instead.
        @available(*, deprecated, message: "Use GenerationOptions custom options instead")
        public var seed: UInt32 { legacyDefaults.seed }

        /// The temperature for sampling.
        ///
        /// - Important: This property is deprecated. Use ``GenerationOptions`` with
        ///   custom options instead.
        @available(*, deprecated, message: "Use GenerationOptions custom options instead")
        public var temperature: Float { legacyDefaults.temperature }

        /// The top-K sampling parameter.
        ///
        /// - Important: This property is deprecated. Use ``GenerationOptions`` with
        ///   custom options instead.
        @available(*, deprecated, message: "Use GenerationOptions custom options instead")
        public var topK: Int32 { legacyDefaults.topK }

        /// The top-P (nucleus) sampling parameter.
        ///
        /// - Important: This property is deprecated. Use ``GenerationOptions`` with
        ///   custom options instead.
        @available(*, deprecated, message: "Use GenerationOptions custom options instead")
        public var topP: Float { legacyDefaults.topP }

        /// The repeat penalty for generation.
        ///
        /// - Important: This property is deprecated. Use ``GenerationOptions`` with
        ///   custom options instead.
        @available(*, deprecated, message: "Use GenerationOptions custom options instead")
        public var repeatPenalty: Float { legacyDefaults.repeatPenalty }

        /// The number of tokens to consider for repeat penalty.
        ///
        /// - Important: This property is deprecated. Use ``GenerationOptions`` with
        ///   custom options instead.
        @available(*, deprecated, message: "Use GenerationOptions custom options instead")
        public var repeatLastN: Int32 { legacyDefaults.repeatLastN }

        /// Normalized legacy defaults used for deprecated properties.
        private let legacyDefaults: ResolvedGenerationOptions

        /// The minimum log level for llama.cpp output.
        ///
        /// This is a global setting that affects all `LlamaLanguageModel` instances
        /// since llama.cpp uses a single global log callback.
        public nonisolated(unsafe) static var logLevel: LogLevel = .warn {
            didSet {
                currentLogLevel = logLevel
                llama_log_set(llamaLogCallback, nil)
            }
        }

        /// Resolved, non-optional defaults for llama.cpp runtime parameters.
        internal struct ResolvedGenerationOptions: Sendable {
            var contextSize: UInt32
            var batchSize: UInt32
            var threads: Int32
            var seed: UInt32
            var temperature: Float
            var topK: Int32
            var topP: Float
            var repeatPenalty: Float
            var repeatLastN: Int32
            var frequencyPenalty: Float
            var presencePenalty: Float
            var mirostat: CustomGenerationOptions.MirostatMode?
            var sampling: GenerationOptions.SamplingMode?
            var maximumResponseTokens: Int?

            init(
                contextSize: UInt32 = 2048,
                batchSize: UInt32 = 512,
                threads: Int32 = Int32(ProcessInfo.processInfo.processorCount),
                seed: UInt32 = UInt32.random(in: 0 ... UInt32.max),
                temperature: Float = 0.8,
                topK: Int32 = 40,
                topP: Float = 0.95,
                repeatPenalty: Float = 1.1,
                repeatLastN: Int32 = 64,
                frequencyPenalty: Float = 0.0,
                presencePenalty: Float = 0.0,
                mirostat: CustomGenerationOptions.MirostatMode? = nil,
                sampling: GenerationOptions.SamplingMode? = nil,
                maximumResponseTokens: Int? = nil
            ) {
                self.contextSize = contextSize
                self.batchSize = batchSize
                self.threads = threads
                self.seed = seed
                self.temperature = temperature
                self.topK = topK
                self.topP = topP
                self.repeatPenalty = repeatPenalty
                self.repeatLastN = repeatLastN
                self.frequencyPenalty = frequencyPenalty
                self.presencePenalty = presencePenalty
                self.mirostat = mirostat
                self.sampling = sampling
                self.maximumResponseTokens = maximumResponseTokens
            }

            init(
                from options: CustomGenerationOptions?,
                sampling: GenerationOptions.SamplingMode? = nil,
                maximumResponseTokens: Int? = nil
            ) {
                self.init(
                    base: ResolvedGenerationOptions(),
                    overrides: options,
                    sampling: sampling,
                    maximumResponseTokens: maximumResponseTokens
                )
            }

            init(
                base: ResolvedGenerationOptions = .init(),
                overrides options: CustomGenerationOptions?,
                sampling: GenerationOptions.SamplingMode? = nil,
                maximumResponseTokens: Int? = nil
            ) {
                guard let options else {
                    self = ResolvedGenerationOptions(
                        contextSize: base.contextSize,
                        batchSize: base.batchSize,
                        threads: base.threads,
                        seed: base.seed,
                        temperature: base.temperature,
                        topK: base.topK,
                        topP: base.topP,
                        repeatPenalty: base.repeatPenalty,
                        repeatLastN: base.repeatLastN,
                        frequencyPenalty: base.frequencyPenalty,
                        presencePenalty: base.presencePenalty,
                        mirostat: base.mirostat,
                        sampling: sampling ?? base.sampling,
                        maximumResponseTokens: maximumResponseTokens ?? base.maximumResponseTokens
                    )
                    return
                }

                self.contextSize = options.contextSize ?? base.contextSize
                self.batchSize = options.batchSize ?? base.batchSize
                self.threads = options.threads ?? base.threads
                self.seed = options.seed ?? base.seed
                self.temperature = options.temperature ?? base.temperature
                self.topK = options.topK ?? base.topK
                self.topP = options.topP ?? base.topP
                self.repeatPenalty = options.repeatPenalty ?? base.repeatPenalty
                self.repeatLastN = options.repeatLastN ?? base.repeatLastN
                self.frequencyPenalty = options.frequencyPenalty ?? base.frequencyPenalty
                self.presencePenalty = options.presencePenalty ?? base.presencePenalty
                self.mirostat = options.mirostat ?? base.mirostat
                self.sampling = sampling ?? base.sampling
                self.maximumResponseTokens = maximumResponseTokens ?? base.maximumResponseTokens
            }
        }

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
        public init(modelPath: String) {
            self.modelPath = modelPath
            self.legacyDefaults = ResolvedGenerationOptions()
        }

        /// Creates a Llama language model using legacy parameter defaults.
        ///
        /// - Important: This initializer is deprecated. Use
        ///   `init(modelPath:defaultOptions:)` and configure per-request values via
        ///   ``GenerationOptions`` custom options instead.
        @available(
            *,
            deprecated,
            message: "Use init(modelPath:) and pass values via GenerationOptions custom options"
        )
        public convenience init(
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
            // Deprecated: prefer setting these via GenerationOptions custom options.
            // We intentionally ignore legacy parameters to avoid storing model-level state.
            self.init(modelPath: modelPath)
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

            // Validate that no image segments are present
            try validateNoImageSegments(in: session)

            try await ensureModelLoaded()

            let runtimeOptions = resolvedOptions(from: options)
            let contextParams = createContextParams(from: runtimeOptions)

            // Try to create context with error handling
            guard let context = llama_init_from_model(model!, contextParams) else {
                throw LlamaLanguageModelError.contextInitializationFailed
            }

            defer { llama_free(context) }

            llama_set_causal_attn(context, true)
            llama_set_warmup(context, false)
            llama_set_n_threads(context, runtimeOptions.threads, runtimeOptions.threads)

            let maxTokens = runtimeOptions.maximumResponseTokens ?? 100
            let text = try await generateText(
                context: context,
                model: model!,
                prompt: prompt.description,
                maxTokens: maxTokens,
                options: runtimeOptions
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

            // Validate that no image segments are present
            do {
                try validateNoImageSegments(in: session)
            } catch {
                return LanguageModelSession.ResponseStream(
                    stream: AsyncThrowingStream { continuation in
                        continuation.finish(throwing: error)
                    }
                )
            }

            let stream: AsyncThrowingStream<LanguageModelSession.ResponseStream<Content>.Snapshot, any Error> =
                AsyncThrowingStream { continuation in
                    let task = Task {
                        do {
                            try await ensureModelLoaded()

                            let runtimeOptions = resolvedOptions(from: options)
                            let maxTokens = runtimeOptions.maximumResponseTokens ?? 100
                            let contextParams = createContextParams(from: runtimeOptions)
                            guard let context = llama_init_from_model(model!, contextParams) else {
                                throw LlamaLanguageModelError.contextInitializationFailed
                            }
                            defer { llama_free(context) }

                            // Stabilize runtime behavior per-context
                            llama_set_causal_attn(context, true)
                            llama_set_warmup(context, false)
                            llama_set_n_threads(context, runtimeOptions.threads, runtimeOptions.threads)

                            var accumulatedText = ""

                            do {
                                for try await tokenText in generateTextStream(
                                    context: context,
                                    model: model!,
                                    prompt: prompt.description,
                                    maxTokens: maxTokens,
                                    options: runtimeOptions
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

        private func resolvedOptions(from options: GenerationOptions) -> ResolvedGenerationOptions {
            var base = legacyDefaults
            if let temp = options.temperature {
                base.temperature = Float(temp)
            }

            return ResolvedGenerationOptions(
                base: base,
                overrides: options[custom: LlamaLanguageModel.self],
                sampling: options.sampling,
                maximumResponseTokens: options.maximumResponseTokens
            )
        }

        private func createContextParams(from options: ResolvedGenerationOptions) -> llama_context_params {
            var params = llama_context_default_params()
            params.n_ctx = options.contextSize
            params.n_batch = options.batchSize
            params.n_threads = options.threads
            params.n_threads_batch = options.threads
            return params
        }

        private func applySampling(
            sampler: UnsafeMutablePointer<llama_sampler>,
            effectiveTemperature: Float,
            options: ResolvedGenerationOptions
        ) {
            if let mirostat = options.mirostat {
                llama_sampler_chain_add(sampler, llama_sampler_init_temp(effectiveTemperature))

                switch mirostat {
                case .v1(let tau, let eta):
                    llama_sampler_chain_add(
                        sampler,
                        llama_sampler_init_mirostat(
                            Int32(options.contextSize),
                            options.seed,
                            tau,
                            eta,
                            100
                        )
                    )
                case .v2(let tau, let eta):
                    llama_sampler_chain_add(sampler, llama_sampler_init_mirostat_v2(options.seed, tau, eta))
                }
                return
            }

            if let sampling = options.sampling {
                switch sampling.mode {
                case .greedy:
                    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(1))
                    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(1.0, 1))
                    llama_sampler_chain_add(sampler, llama_sampler_init_greedy())
                case .topK(let k, let seed):
                    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(Int32(k)))
                    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(1.0, 1))
                    llama_sampler_chain_add(sampler, llama_sampler_init_temp(effectiveTemperature))
                    let samplingSeed = seed.map(UInt32.init) ?? options.seed
                    llama_sampler_chain_add(sampler, llama_sampler_init_dist(samplingSeed))
                case .nucleus(let threshold, let seed):
                    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(0))
                    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(Float(threshold), 1))
                    llama_sampler_chain_add(sampler, llama_sampler_init_temp(effectiveTemperature))
                    let samplingSeed = seed.map(UInt32.init) ?? options.seed
                    llama_sampler_chain_add(sampler, llama_sampler_init_dist(samplingSeed))
                }
                return
            }

            if options.topK > 0 {
                llama_sampler_chain_add(sampler, llama_sampler_init_top_k(options.topK))
            }
            if options.topP < 1.0 {
                llama_sampler_chain_add(sampler, llama_sampler_init_top_p(options.topP, 1))
            }
            llama_sampler_chain_add(sampler, llama_sampler_init_temp(effectiveTemperature))
            llama_sampler_chain_add(sampler, llama_sampler_init_dist(options.seed))
        }

        private func generateText(
            context: OpaquePointer,
            model: OpaquePointer,
            prompt: String,
            maxTokens: Int,
            options: ResolvedGenerationOptions
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

            var batch = llama_batch_init(Int32(options.batchSize), 0, 1)
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
            let samplerPtr = UnsafeMutablePointer<llama_sampler>(sampler)

            let effectiveTemperature = Float(options.temperature)

            // Apply repeat/frequency/presence penalties from custom options
            let effectiveRepeatPenalty = options.repeatPenalty
            let effectiveRepeatLastN = options.repeatLastN
            let effectiveFrequencyPenalty = options.frequencyPenalty
            let effectivePresencePenalty = options.presencePenalty

            if effectiveRepeatPenalty != 1.0 || effectiveFrequencyPenalty != 0.0 || effectivePresencePenalty != 0.0 {
                llama_sampler_chain_add(
                    samplerPtr,
                    llama_sampler_init_penalties(
                        effectiveRepeatLastN,
                        effectiveRepeatPenalty,
                        effectiveFrequencyPenalty,
                        effectivePresencePenalty
                    )
                )
            }

            applySampling(sampler: samplerPtr, effectiveTemperature: effectiveTemperature, options: options)

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
            options: ResolvedGenerationOptions
        ) -> AsyncThrowingStream<String, Error> {
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
            options: ResolvedGenerationOptions,
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
                var batch = llama_batch_init(Int32(options.batchSize), 0, 1)
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
                let samplerPtr = UnsafeMutablePointer<llama_sampler>(sampler)

                let effectiveTemperature = Float(options.temperature)

                // Apply repeat/frequency/presence penalties from custom options
                let effectiveRepeatPenalty = options.repeatPenalty
                let effectiveRepeatLastN = options.repeatLastN
                let effectiveFrequencyPenalty = options.frequencyPenalty
                let effectivePresencePenalty = options.presencePenalty

                if effectiveRepeatPenalty != 1.0 || effectiveFrequencyPenalty != 0.0 || effectivePresencePenalty != 0.0
                {
                    llama_sampler_chain_add(
                        samplerPtr,
                        llama_sampler_init_penalties(
                            effectiveRepeatLastN,
                            effectiveRepeatPenalty,
                            effectiveFrequencyPenalty,
                            effectivePresencePenalty
                        )
                    )
                }

                // Check for mirostat sampling (takes precedence over standard sampling)
                applySampling(sampler: samplerPtr, effectiveTemperature: effectiveTemperature, options: options)

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

        // MARK: - Image Validation

        private func validateNoImageSegments(in session: LanguageModelSession) throws {
            // Check for image segments in the most recent prompt from the transcript
            for entry in session.transcript.reversed() {
                if case .prompt(let p) = entry {
                    for segment in p.segments {
                        if case .image = segment {
                            throw LlamaLanguageModelError.unsupportedFeature
                        }
                    }
                    break
                }
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
