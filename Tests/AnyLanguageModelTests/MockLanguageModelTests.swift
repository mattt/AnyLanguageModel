import Testing

@testable import AnyLanguageModel

@Suite("MockLanguageModel")
struct MockLanguageModelTests {
    @Test func fixed() async throws {
        let model = MockLanguageModel.fixed("Hello, World!")
        let session = LanguageModelSession(model: model)

        let response = try await session.respond(to: Prompt("Test prompt"))
        #expect(response.content == "Hello, World!")
    }

    @Test func echo() async throws {
        let model = MockLanguageModel.echo
        let session = LanguageModelSession(model: model)

        let prompt = Prompt("Echo this")
        let response = try await session.respond(to: prompt)
        #expect(response.content.contains("Echo this"))
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

            let response = try await session.respond(to: Prompt("Test"))
            #expect(response.content == expected)
        }
    }
}
