import Foundation
import Testing

@testable import AnyLanguageModel

@Suite("OllamaLanguageModel")
struct OllamaLanguageModelTests {
    let model = OllamaLanguageModel(model: "qwen3:8b")

    @Test func customHost() {
        let customURL = URL(string: "http://example.com")!
        let model = OllamaLanguageModel(baseURL: customURL, model: "custom")
        #expect(model.baseURL.absoluteString.hasSuffix("/"))
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

    @Test func conversationContext() async throws {
        let session = LanguageModelSession(model: model)

        let firstResponse = try await session.respond(to: "My favorite color is blue")
        #expect(!firstResponse.content.isEmpty)

        let secondResponse = try await session.respond(to: "What did I just tell you?")
        #expect(!secondResponse.content.isEmpty)
    }

    @Test func withTools() async throws {
        let weatherTool = spy(on: WeatherTool())
        let session = LanguageModelSession(model: model, tools: [weatherTool])

        let response = try await session.respond(to: "What's the weather in San Francisco?")

        var foundToolOutput = false
        for case let .toolOutput(toolOutput) in response.transcriptEntries {
            #expect(toolOutput.id == weatherTool.name)
            foundToolOutput = true
        }
        #expect(foundToolOutput)

        let calls = await weatherTool.calls
        #expect(calls.count == 1)
        #expect(calls.first?.arguments.city == "San Francisco")

        if case .success(let output) = calls.first?.result {
            #expect(output.contains("San Francisco"))
        } else {
            Issue.record("Expected successful tool call")
        }
    }
}
