@testable import AnyLanguageModel

struct MockLanguageModel: LanguageModel {
    enum UnavailableReason: Hashable, Sendable {
        case custom(String)
    }

    var availabilityProvider: @Sendable () -> Availability<UnavailableReason>
    var responseProvider: @Sendable (Prompt, GenerationOptions) async throws -> String

    init(
        _ responseProvider:
            @escaping @Sendable (Prompt, GenerationOptions) async throws ->
            String = { _, _ in "Mock response" }
    ) {
        self.availabilityProvider = { .available }
        self.responseProvider = responseProvider
    }

    var availability: Availability<UnavailableReason> {
        return availabilityProvider()
    }

    func respond<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
        // For now, only String is supported
        guard type == String.self else {
            fatalError("MockLanguageModel only supports generating String content")
        }

        let promptWithInstructions = Prompt("Instructions: \(session.instructions?.description ?? "N/A")\n\(prompt))")
        let text = try await responseProvider(promptWithInstructions, options)

        return LanguageModelSession.Response(
            content: text as! Content,
            rawContent: GeneratedContent(text),
            transcriptEntries: []
        )
    }

    func streamResponse<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) -> sending LanguageModelSession.ResponseStream<Content> where Content: Generable {
        // For now, only String is supported
        guard type == String.self else {
            fatalError("MockLanguageModel only supports generating String content")
        }

        // For MockLanguageModel, we'll simulate streaming by yielding the response immediately
        // In a real implementation, this would stream the response as it's generated
        // Since we can't make this function async, we'll need to handle this differently
        // For now, we'll create a stream that yields immediately with a placeholder
        let placeholderText = "Mock streaming response"
        let generatedContent = GeneratedContent(placeholderText)

        return LanguageModelSession.ResponseStream(content: placeholderText as! Content, rawContent: generatedContent)
    }
}

// MARK: -

extension MockLanguageModel {
    static var echo: Self {
        MockLanguageModel { prompt, _ in
            prompt.description
        }
    }

    static func fixed(_ response: String) -> Self {
        MockLanguageModel { _, _ in response }
    }

    static var unavailable: Self {
        var model = MockLanguageModel()
        model.availabilityProvider = { .unavailable(.custom("MockLanguageModel is unavailable")) }
        return model
    }
}
