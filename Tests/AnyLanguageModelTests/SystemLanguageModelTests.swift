import Testing

@testable import AnyLanguageModel

#if canImport(FoundationModels)
    private let isSystemLanguageModelAvailable = {
        if #available(macOS 26.0, *) {
            return SystemLanguageModel().systemModel.isAvailable
        } else {
            return false
        }
    }()

    @Suite("SystemLanguageModel", .enabled(if: isSystemLanguageModelAvailable))
    struct SystemLanguageModelTests {
        @available(macOS 26.0, *)
        @Test func basicResponse() async throws {
            let model: SystemLanguageModel = SystemLanguageModel()
            let session = LanguageModelSession(model: model)

            let response = try await session.respond(to: "Say 'Hello'")
            #expect(!response.content.isEmpty)
        }

        @available(macOS 26.0, *)
        @Test func withInstructions() async throws {
            let model = SystemLanguageModel()
            let session = LanguageModelSession(
                model: model,
                instructions: "You are a helpful assistant."
            )

            let response = try await session.respond(to: "What is 2+2?")
            #expect(!response.content.isEmpty)
        }

        @available(macOS 26.0, *)
        @Test func withTemperature() async throws {
            let model: SystemLanguageModel = SystemLanguageModel()
            let session = LanguageModelSession(model: model)

            let options = GenerationOptions(temperature: 0.5)
            let response = try await session.respond(
                to: "Generate a number",
                options: options
            )
            #expect(!response.content.isEmpty)
        }

        // @available(macOS 26.0, *)
        // @Test func streamResponse() async throws {
        //     let model: SystemLanguageModel = SystemLanguageModel()
        //     let session = LanguageModelSession(model: model)

        //     let stream = session.streamResponse(to: "Count to 3")

        //     var responses: [Response<String>] = []
        //     for try await response in stream {
        //         responses.append(response)
        //     }

        //     #expect(!responses.isEmpty)
        //     #expect(!responses.last!.text.isEmpty)
        // }

        // @available(macOS 26.0, *)
        // @Test func streamWithInstructions() async throws {
        //     let model = SystemLanguageModel()
        //     let session = LanguageModelSession(
        //         model: model,
        //         instructions: "Be concise."
        //     )

        //     let stream = try await session.streamResponse(to: "Say hi")

        //     var responses: [Response] = []
        //     for try await response in stream {
        //         responses.append(response)
        //     }

        //     #expect(!responses.isEmpty)
        // }
    }
#endif
