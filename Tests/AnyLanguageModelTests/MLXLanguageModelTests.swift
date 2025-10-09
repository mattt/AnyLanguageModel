import Foundation
import Testing

@testable import AnyLanguageModel

// Check if Metal/MLX is available
private let isMLXAvailable = {
    // Skip in CI environments
    if ProcessInfo.processInfo.environment["CI"] != nil {
        return false
    }

    // Enable when running with Xcode/xcodebuild
    if ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil {
        return true
    }

    // Enable when explicitly requested via environment variable
    if ProcessInfo.processInfo.environment["ENABLE_MLX_TESTS"] != nil {
        return true
    }

    // Skip by default when running with swift test
    return false
}()

@Suite("MLXLanguageModel", .enabled(if: isMLXAvailable))
struct MLXLanguageModelTests {
    let model = MLXLanguageModel(modelId: "mlx-community/Qwen1.5-0.5B-Chat-4bit")

    @Test func basicResponse() async throws {
        let session = LanguageModelSession(model: model)

        let response = try await session.respond(to: Prompt("Say hello"))
        #expect(!response.content.isEmpty)
    }

    @Test func withGenerationOptions() async throws {
        let session = LanguageModelSession(model: model)

        let options = GenerationOptions(
            temperature: 0.7,
            maximumResponseTokens: 32
        )

        let response = try await session.respond(
            to: Prompt("Tell me a fact"),
            options: options
        )
        #expect(!response.content.isEmpty)
    }

    @Test func withTools() async throws {
        let weatherTool = spy(on: WeatherTool())
        let session = LanguageModelSession(model: model, tools: [weatherTool])

        // Prompt that encourages tool usage per MLXLMCommon ToolCallProcessor
        // Expected format: <tool_call>{"name":"getWeather","arguments":{...}}</tool_call>
        let response = try await session.respond(
            to: Prompt("Use provided tools. What's the weather in San Francisco?")
        )

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
}
