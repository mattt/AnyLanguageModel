import Testing

#if canImport(FoundationModels)
    import AnyLanguageModel

    @available(macOS 26.0, *)
    @Test("AnyLanguageModel Drop-In Compatibility", .enabled(if: SystemLanguageModel.default.isAvailable))
    func anyLanguageModelCompatibility() async throws {
        let model = SystemLanguageModel.default
        let session = LanguageModelSession(
            model: model,
            instructions: Instructions("You are a helpful assistant.")
        )

        let options = GenerationOptions(temperature: 0.7)
        let response = try await session.respond(options: options) {
            Prompt("Say 'Hello'")
        }
        #expect(!response.content.isEmpty)

        let stream = session.streamResponse {
            Prompt("Count to 3")
        }
        var hasSnapshots = false
        for try await _ in stream {
            hasSnapshots = true
            break
        }
        #expect(hasSnapshots)
    }
#endif
