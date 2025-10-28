import Testing

@testable import AnyLanguageModel

@Suite("MockLanguageModel")
struct MockLanguageModelTests {
    @Test func fixedResponse() async throws {
        let model = MockLanguageModel.fixed("Hello, World!")
        let session = LanguageModelSession(model: model)

        #expect(session.transcript.count == 0)

        let response = try await session.respond(to: "Say hello")
        #expect(response.content == "Hello, World!")

        // Verify transcript was updated
        #expect(session.transcript.count == 2)
        #expect(response.transcriptEntries.count > 0)
    }

    @Test func echoResponse() async throws {
        let model = MockLanguageModel.echo
        let session = LanguageModelSession(model: model)

        let prompt = Prompt("Echo this")
        let response = try await session.respond(to: prompt)
        #expect(response.content.contains(prompt.description))

        // Verify transcript
        #expect(session.transcript.count == 2)
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

        for (instructionText, expected) in [
            ("Be helpful", "😇"),
            ("Be evil", "😈"),
            ("Meh", "😐"),
        ] {
            let session = LanguageModelSession(
                model: model,
                instructions: Instructions(instructionText)
            )

            // Verify instructions are in transcript
            let entriesBeforeResponse = Array(session.transcript)
            #expect(entriesBeforeResponse.count == 1)
            if case .instructions(let transcriptInstructions) = entriesBeforeResponse.first {
                #expect(transcriptInstructions.segments.count > 0)
            } else {
                Issue.record("First entry should be instructions")
            }

            let response = try await session.respond(to: "Do what you want")
            #expect(response.content == expected)

            // Verify transcript has instructions, prompt, and response
            #expect(session.transcript.count == 3)
        }
    }

    @Test func unavailable() async throws {
        let model = MockLanguageModel.unavailable

        #expect(model.availability == .unavailable(.custom("MockLanguageModel is unavailable")))
        #expect(model.isAvailable == false)
    }

    @Test func streamingResponse() async throws {
        // Test async response with isResponding state
        let asyncModel = MockLanguageModel { _, _ in
            try await Task.sleep(for: .milliseconds(100))
            return "Async Response"
        }
        let asyncSession = LanguageModelSession(model: asyncModel)

        #expect(asyncSession.isResponding == false)
        #expect(asyncSession.transcript.count == 0)

        let asyncTask = Task {
            try await asyncSession.respond(to: "Async test")
        }

        try await Task.sleep(for: .milliseconds(50))
        #expect(asyncSession.isResponding == true)

        let response = try await asyncTask.value
        try await Task.sleep(for: .milliseconds(10))
        #expect(asyncSession.isResponding == false)
        #expect(asyncSession.transcript.count == 2)
        #expect(response.transcriptEntries.count > 0)

        // Test streaming response with isResponding state
        let streamModel = MockLanguageModel.streamingMock()
        let streamSession = LanguageModelSession(model: streamModel)

        #expect(streamSession.isResponding == false)
        #expect(streamSession.transcript.count == 0)

        let stream = streamSession.streamResponse(to: "Stream test")

        let streamTask = Task {
            for try await _ in stream {}
        }

        try await Task.sleep(for: .milliseconds(50))
        #expect(streamSession.isResponding == true)

        _ = try await streamTask.value
        try await Task.sleep(for: .milliseconds(10))
        #expect(streamSession.isResponding == false)
        #expect(streamSession.transcript.count == 2)
    }

    @Test func transcriptGrowsWithMultipleInteractions() async throws {
        let model = MockLanguageModel.echo
        let session = LanguageModelSession(model: model)

        #expect(session.transcript.count == 0)

        try await session.respond(to: "First prompt")
        let countAfterFirst = session.transcript.count
        #expect(countAfterFirst == 2)

        try await session.respond(to: "Second prompt")
        let countAfterSecond = session.transcript.count
        #expect(countAfterSecond == 4)

        try await session.respond(to: "Third prompt")
        let countAfterThird = session.transcript.count
        #expect(countAfterThird == 6)

        // Verify all entries are identifiable
        for entry in session.transcript {
            #expect(!entry.id.isEmpty)
        }
    }
}
