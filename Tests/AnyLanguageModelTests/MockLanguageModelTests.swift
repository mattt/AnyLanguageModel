import Testing

@testable import AnyLanguageModel

@Suite("MockLanguageModel")
struct MockLanguageModelTests {
    @Test func fixed() async throws {
        let model = MockLanguageModel.fixed("Hello, World!")
        let session = LanguageModelSession(model: model)

        let response = try await session.respond(to: "Say hello")
        #expect(response.content == "Hello, World!")
    }

    @Test func echo() async throws {
        let model = MockLanguageModel.echo
        let session = LanguageModelSession(model: model)

        let prompt = Prompt("Echo this")
        let response = try await session.respond(to: prompt)
        #expect(response.content.contains(prompt.description))
    }

    @Test func withInstructions() async throws {
        let model = MockLanguageModel { prompt, _ in
            if prompt.description.contains("Be helpful") {
                return "😇"
            }

            if prompt.description.contains("Be evil") {
                return "😈"
            }

            return "😐"
        }

        for (prompt, expected) in [
            ("Be helpful", "😇"),
            ("Be evil", "😈"),
            ("Meh", "😐"),
        ] {
            let session = LanguageModelSession(
                model: model,
                instructions: Instructions(prompt)
            )

            let response = try await session.respond(to: "Do what you want")
            #expect(response.content == expected)
        }
    }

    @Test func unavailable() async throws {
        let model = MockLanguageModel.unavailable

        #expect(model.availability == .unavailable(.custom("MockLanguageModel is unavailable")))
        #expect(model.isAvailable == false)
    }

    @Test func isRespondingDuringAsyncResponse() async throws {
        let model = MockLanguageModel { _, _ in
            try await Task.sleep(for: .milliseconds(100))
            return "Response"
        }
        let session = LanguageModelSession(model: model)

        #expect(session.isResponding == false)

        let task = Task {
            try await session.respond(to: "Test")
        }

        try await Task.sleep(for: .milliseconds(50))
        #expect(session.isResponding == true)

        _ = try await task.value
        try await Task.sleep(for: .milliseconds(10))
        #expect(session.isResponding == false)
    }

    @Test func isRespondingDuringStreaming() async throws {
        let model = MockLanguageModel.streamingMock()
        let session = LanguageModelSession(model: model)

        #expect(session.isResponding == false)

        let stream = session.streamResponse(to: "Test")

        // Start consuming the stream in a task
        let task = Task {
            for try await _ in stream {
                // Just consume the stream
            }
        }

        // Give the streaming task time to start and call beginResponding
        try await Task.sleep(for: .milliseconds(50))
        #expect(session.isResponding == true)

        // Wait for stream to complete
        _ = try await task.value

        // Give time for endResponding to complete
        try await Task.sleep(for: .milliseconds(10))
        #expect(session.isResponding == false)
    }
}
