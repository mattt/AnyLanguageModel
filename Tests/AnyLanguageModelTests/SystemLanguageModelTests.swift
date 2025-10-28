import Testing
import AnyLanguageModel

#if canImport(FoundationModels)
    private let isSystemLanguageModelAvailable = {
        if #available(macOS 26.0, *) {
            return SystemLanguageModel.default.isAvailable
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

        @available(macOS 26.0, *)
        @Test func streamingString() async throws {
            guard isSystemLanguageModelAvailable else { return }
            let model: SystemLanguageModel = SystemLanguageModel()
            let session = LanguageModelSession(model: model)

            let stream = session.streamResponse(to: "Say 'Hello' slowly")

            var snapshots: [LanguageModelSession.ResponseStream<String>.Snapshot] = []
            for try await snapshot in stream {
                snapshots.append(snapshot)
            }

            #expect(!snapshots.isEmpty)
            #expect(!snapshots.last!.rawContent.jsonString.isEmpty)
        }

        @available(macOS 26.0, *)
        @Test func streamingGeneratedContent() async throws {
            guard isSystemLanguageModelAvailable else { return }
            let model: SystemLanguageModel = SystemLanguageModel()
            let session = LanguageModelSession(model: model)

            let stream = session.streamResponse(
                to: Prompt("Provide a JSON object with a field 'text'"),
                schema: GeneratedContent.generationSchema
            )

            var snapshots: [LanguageModelSession.ResponseStream<GeneratedContent>.Snapshot] = []
            for try await snapshot in stream {
                snapshots.append(snapshot)
            }

            #expect(!snapshots.isEmpty)
            #expect(!snapshots.last!.rawContent.jsonString.isEmpty)
        }
    }
#endif
