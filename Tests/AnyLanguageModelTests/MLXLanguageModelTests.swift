import Foundation
import Testing

@testable import AnyLanguageModel

#if MLX
    private let shouldRunMLXTests = {
        // Enable when explicitly requested via environment variable
        if ProcessInfo.processInfo.environment["ENABLE_MLX_TESTS"] != nil {
            return true
        }

        // Skip in CI environments
        if ProcessInfo.processInfo.environment["CI"] != nil {
            return false
        }

        // Skip unless Hugging Face API token is provided
        if ProcessInfo.processInfo.environment["HF_TOKEN"] == nil {
            return false
        }

        // Enable when running with Xcode/xcodebuild
        if ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil {
            return true
        }

        // Skip by default when running with swift test
        return false
    }()

    @Suite("MLXLanguageModel", .enabled(if: shouldRunMLXTests), .serialized)
    struct MLXLanguageModelTests {
        // Qwen3-0.6B is a small model that supports tool calling
        let model = MLXLanguageModel(modelId: "mlx-community/Qwen3-0.6B-4bit")
        let visionModel = MLXLanguageModel(modelId: "mlx-community/Qwen2-VL-2B-Instruct-4bit")

        @Test func basicResponse() async throws {
            let session = LanguageModelSession(model: model)

            let response = try await session.respond(to: "Say hello")
            #expect(!response.content.isEmpty)
        }

        @Test func streamingResponse() async throws {
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
                maximumResponseTokens: 32
            )

            let response = try await session.respond(
                to: "Tell me a fact",
                options: options
            )
            #expect(!response.content.isEmpty)
        }

        @Test func withTools() async throws {
            let weatherTool = spy(on: WeatherTool())
            let session = LanguageModelSession(
                model: model,
                tools: [weatherTool],
                instructions: "You are a helpful assistant. Use available tools when needed."
            )

            let response = try await session.respond(to: "How's the weather in San Francisco?")

            var foundToolOutput = false
            for case let .toolOutput(toolOutput) in response.transcriptEntries {
                #expect(toolOutput.id == weatherTool.name)
                foundToolOutput = true
            }
            #expect(foundToolOutput)

            let calls = await weatherTool.calls
            #expect(calls.count >= 1)
            if let first = calls.first {
                #expect(first.arguments.city.contains("San Francisco"))
            }
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
            let session = LanguageModelSession(model: visionModel, transcript: transcript)
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
            let session = LanguageModelSession(model: visionModel, transcript: transcript)
            let response = try await session.respond(to: "")
            #expect(!response.content.isEmpty)
        }
    }
#endif  // MLX
