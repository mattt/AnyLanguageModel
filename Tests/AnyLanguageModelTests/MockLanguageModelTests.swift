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

    @Test func transcriptStartsEmpty() async throws {
        let model = MockLanguageModel.fixed("Hello")
        let session = LanguageModelSession(model: model)

        #expect(session.transcript.count == 0)
    }

    @Test func transcriptRecordsPromptAndResponse() async throws {
        let model = MockLanguageModel.fixed("Hello, World!")
        let session = LanguageModelSession(model: model)

        #expect(session.transcript.count == 0)

        let response = try await session.respond(to: "Say hello")

        let entries = Array(session.transcript)
        #expect(entries.count > 0)
        #expect(response.transcriptEntries.count > 0)

        #expect(response.content == "Hello, World!")
    }

    @Test func transcriptGrowsWithMultipleInteractions() async throws {
        let model = MockLanguageModel.echo
        let session = LanguageModelSession(model: model)

        #expect(session.transcript.count == 0)

        try await session.respond(to: "First prompt")
        let countAfterFirst = session.transcript.count
        #expect(countAfterFirst > 0)

        try await session.respond(to: "Second prompt")
        let countAfterSecond = session.transcript.count
        #expect(countAfterSecond > countAfterFirst)

        try await session.respond(to: "Third prompt")
        let countAfterThird = session.transcript.count
        #expect(countAfterThird > countAfterSecond)
    }

    @Test func transcriptIncludesInstructions() async throws {
        let model = MockLanguageModel.fixed("Response")
        let instructions = Instructions("Be helpful")
        let session = LanguageModelSession(
            model: model,
            instructions: instructions
        )

        let entries = Array(session.transcript)
        #expect(entries.count > 0)

        if case .instructions(let transcriptInstructions) = entries.first {
            #expect(transcriptInstructions.segments.count > 0)
        } else {
            Issue.record("First transcript entry should be instructions")
        }
    }

    @Test func transcriptEntriesAreIdentifiable() async throws {
        let model = MockLanguageModel.fixed("Response")
        let session = LanguageModelSession(model: model)

        try await session.respond(to: "Test")

        let entries = Array(session.transcript)
        #expect(entries.count > 0)

        for entry in entries {
            #expect(!entry.id.isEmpty)
        }
    }

    @Test func streamingRecordsToTranscript() async throws {
        let model = MockLanguageModel.fixed("Streamed response")
        let session = LanguageModelSession(model: model)

        #expect(session.transcript.count == 0)

        let stream = session.streamResponse(to: "Stream this")

        // Consume the stream
        for try await _ in stream {}

        // Give time for transcript update to complete
        try await Task.sleep(for: .milliseconds(10))

        // Verify transcript has both prompt and response
        let entries = Array(session.transcript)
        #expect(entries.count == 2)

        // First entry should be prompt
        if case .prompt(let prompt) = entries[0] {
            #expect(prompt.segments.count > 0)
        } else {
            Issue.record("First entry should be prompt")
        }

        // Second entry should be response
        if case .response(let response) = entries[1] {
            #expect(response.segments.count > 0)
        } else {
            Issue.record("Second entry should be response")
        }
    }
}
