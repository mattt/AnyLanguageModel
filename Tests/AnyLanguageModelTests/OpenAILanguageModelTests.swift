import Foundation
import Testing

@testable import AnyLanguageModel

private let openaiAPIKey: String? = ProcessInfo.processInfo.environment["OPENAI_API_KEY"]

@Suite("OpenAILanguageModel")
struct OpenAILanguageModelTests {
    @Test func customHost() throws {
        let customURL = URL(string: "https://example.com")!
        let model = OpenAILanguageModel(baseURL: customURL, apiKey: "test", model: "test-model")
        #expect(model.baseURL.absoluteString.hasSuffix("/"))
    }

    @Test func apiVariantParameterization() throws {
        // Test that both API variants can be created and have correct properties
        for apiVariant in [OpenAILanguageModel.APIVariant.chatCompletions, .responses] {
            let model = OpenAILanguageModel(apiKey: "test-key", model: "test-model", apiVariant: apiVariant)
            #expect(model.apiVariant == apiVariant)
            #expect(model.model == "test-model")
        }
    }

    @Suite("OpenAILanguageModel Chat Completions API", .enabled(if: openaiAPIKey?.isEmpty == false))
    struct ChatCompletionsTests {
        private let apiKey = openaiAPIKey!

        private var model: OpenAILanguageModel {
            OpenAILanguageModel(apiKey: apiKey, model: "gpt-4o-mini", apiVariant: .chatCompletions)
        }

        @Test func basicResponse() async throws {
            let session = LanguageModelSession(model: model)

            let response = try await session.respond(to: "Say hello")
            #expect(!response.content.isEmpty)
        }

        @Test func withInstructions() async throws {
            let session = LanguageModelSession(
                model: model,
                instructions: "You are a helpful assistant. Be concise."
            )

            let response = try await session.respond(to: "What is 2+2?")
            #expect(!response.content.isEmpty)
        }

        @Test func streaming() async throws {
            let session = LanguageModelSession(model: model)

            let stream = session.streamResponse(to: "Count to 5")
            var chunks: [String] = []

            for try await response in stream {
                chunks.append(response.content)
            }

            #expect(!chunks.isEmpty)
        }

        @Test func streamingString() async throws {
            let session = LanguageModelSession(model: model)

            let stream = session.streamResponse(to: "Say 'Hello' slowly")

            var snapshots: [LanguageModelSession.ResponseStream<String>.Snapshot] = []
            for try await snapshot in stream {
                snapshots.append(snapshot)
            }

            #expect(!snapshots.isEmpty)
            #expect(!snapshots.last!.rawContent.jsonString.isEmpty)
        }

        @Test func withGenerationOptions() async throws {
            let session = LanguageModelSession(model: model)

            let options = GenerationOptions(
                temperature: 0.7,
                maximumResponseTokens: 50
            )

            let response = try await session.respond(
                to: "Tell me a fact",
                options: options
            )
            #expect(!response.content.isEmpty)
        }

        @Test func withCustomGenerationOptions() async throws {
            let session = LanguageModelSession(model: model)

            var options = GenerationOptions(
                temperature: 0.7,
                maximumResponseTokens: 50
            )

            // Set custom options (extraBody will be merged into the request)
            options[custom: OpenAILanguageModel.self] = .init(
                extraBody: ["user": .string("test-user-id")]
            )

            let response = try await session.respond(
                to: "Say hello",
                options: options
            )
            #expect(!response.content.isEmpty)
        }

        @Test func multimodalWithImageURL() async throws {
            let transcript = Transcript(entries: [
                .prompt(
                    Transcript.Prompt(segments: [
                        .text(.init(content: "Describe this image")),
                        .image(.init(url: testImageURL)),
                    ])
                )
            ])
            let session = LanguageModelSession(model: model, transcript: transcript)
            let response = try await session.respond(to: "")
            #expect(!response.content.isEmpty)
        }

        @Test func multimodalWithImageData() async throws {
            let transcript = Transcript(entries: [
                .prompt(
                    Transcript.Prompt(segments: [
                        .text(.init(content: "Describe this image")),
                        .image(.init(data: testImageData, mimeType: "image/png")),
                    ])
                )
            ])
            let session = LanguageModelSession(model: model, transcript: transcript)
            let response = try await session.respond(to: "")
            #expect(!response.content.isEmpty)
        }

        @Test func conversationContext() async throws {
            let session = LanguageModelSession(model: model)

            let firstResponse = try await session.respond(to: "My favorite color is blue")
            #expect(!firstResponse.content.isEmpty)

            let secondResponse = try await session.respond(to: "What did I just tell you?")
            #expect(!secondResponse.content.isEmpty)
        }

        @Test func withTools() async throws {
            let weatherTool = WeatherTool()
            let session = LanguageModelSession(model: model, tools: [weatherTool])

            let response = try await session.respond(to: "How's the weather in San Francisco?")

            var foundToolOutput = false
            for case let .toolOutput(toolOutput) in response.transcriptEntries {
                #expect(toolOutput.toolName == "getWeather")
                foundToolOutput = true
            }
            #expect(foundToolOutput)
        }
    }

    @Suite("OpenAILanguageModel Responses API", .enabled(if: openaiAPIKey?.isEmpty == false))
    struct ResponsesTests {
        private let apiKey = openaiAPIKey!

        private var model: OpenAILanguageModel {
            OpenAILanguageModel(apiKey: apiKey, model: "gpt-4o-mini", apiVariant: .responses)
        }

        @Test func basicResponse() async throws {
            let session = LanguageModelSession(model: model)

            let response = try await session.respond(to: "Say hello")
            #expect(!response.content.isEmpty)
        }

        @Test func withInstructions() async throws {
            let session = LanguageModelSession(
                model: model,
                instructions: "You are a helpful assistant. Be concise."
            )

            let response = try await session.respond(to: "What is 2+2?")
            #expect(!response.content.isEmpty)
        }

        @Test func streaming() async throws {
            let session = LanguageModelSession(model: model)

            let stream = session.streamResponse(to: "Count to 5")
            var chunks: [String] = []

            for try await response in stream {
                chunks.append(response.content)
            }

            #expect(!chunks.isEmpty)
        }

        @Test func streamingString() async throws {
            let session = LanguageModelSession(model: model)

            let stream = session.streamResponse(to: "Say 'Hello' slowly")

            var snapshots: [LanguageModelSession.ResponseStream<String>.Snapshot] = []
            for try await snapshot in stream {
                snapshots.append(snapshot)
            }

            #expect(!snapshots.isEmpty)
            #expect(!snapshots.last!.rawContent.jsonString.isEmpty)
        }

        @Test func withGenerationOptions() async throws {
            let session = LanguageModelSession(model: model)

            let options = GenerationOptions(
                temperature: 0.7,
                maximumResponseTokens: 50
            )

            let response = try await session.respond(
                to: "Tell me a fact",
                options: options
            )
            #expect(!response.content.isEmpty)
        }

        @Test func withCustomGenerationOptions() async throws {
            let session = LanguageModelSession(model: model)

            var options = GenerationOptions(
                temperature: 0.7,
                maximumResponseTokens: 50
            )

            // Set custom options (extraBody will be merged into the request)
            options[custom: OpenAILanguageModel.self] = .init(
                extraBody: ["user": "test-user-id"]
            )

            let response = try await session.respond(
                to: "Say hello",
                options: options
            )
            #expect(!response.content.isEmpty)
        }

        @Test func multimodalWithImageURL() async throws {
            let transcript = Transcript(entries: [
                .prompt(
                    Transcript.Prompt(segments: [
                        .text(.init(content: "Describe this image")),
                        .image(.init(url: testImageURL)),
                    ])
                )
            ])
            let session = LanguageModelSession(model: model, transcript: transcript)
            let response = try await session.respond(to: "")
            #expect(!response.content.isEmpty)
        }

        @Test func multimodalWithImageData() async throws {
            let transcript = Transcript(entries: [
                .prompt(
                    Transcript.Prompt(segments: [
                        .text(.init(content: "Describe this image")),
                        .image(.init(data: testImageData, mimeType: "image/png")),
                    ])
                )
            ])
            let session = LanguageModelSession(model: model, transcript: transcript)
            let response = try await session.respond(to: "")
            #expect(!response.content.isEmpty)
        }

        @Test func conversationContext() async throws {
            let session = LanguageModelSession(model: model)

            let firstResponse = try await session.respond(to: "My favorite color is blue")
            #expect(!firstResponse.content.isEmpty)

            let secondResponse = try await session.respond(to: "What did I just tell you?")
            #expect(!secondResponse.content.isEmpty)
        }

        @Test func withTools() async throws {
            let weatherTool = WeatherTool()
            let session = LanguageModelSession(model: model, tools: [weatherTool])

            let response = try await session.respond(to: "How's the weather in San Francisco?")

            var foundToolOutput = false
            for case let .toolOutput(toolOutput) in response.transcriptEntries {
                #expect(toolOutput.toolName == "getWeather")
                foundToolOutput = true
            }
            #expect(foundToolOutput)
        }
    }
}
