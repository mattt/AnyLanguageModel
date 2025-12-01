import Foundation
import Testing

@testable import AnyLanguageModel

#if Llama
    @Suite(
        "LlamaLanguageModel",
        .serialized,
        .enabled(if: ProcessInfo.processInfo.environment["LLAMA_MODEL_PATH"] != nil)
    )
    struct LlamaLanguageModelTests {
        let modelPath: String = ProcessInfo.processInfo.environment["LLAMA_MODEL_PATH"] ?? ""

        var model: LlamaLanguageModel {
            LlamaLanguageModel(
                modelPath: modelPath,
                contextSize: 2048,
                temperature: 0.8
            )
        }

        @Test func initialization() {
            let customModel = LlamaLanguageModel(
                modelPath: "/path/to/model.gguf",
                contextSize: 4096,
                temperature: 0.7,
                topK: 50
            )
            #expect(customModel.modelPath == "/path/to/model.gguf")
            #expect(customModel.contextSize == 4096)
            #expect(customModel.temperature == 0.7)
            #expect(customModel.topK == 50)
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

        @Test func conversationContext() async throws {
            let session = LanguageModelSession(model: model)

            let firstResponse = try await session.respond(to: "My favorite color is blue")
            #expect(!firstResponse.content.isEmpty)

            let secondResponse = try await session.respond(to: "What did I just tell you?")
            #expect(!secondResponse.content.isEmpty)
        }

        @Test func customTemperature() async throws {
            let highTempModel = LlamaLanguageModel(
                modelPath: modelPath,
                temperature: 1.5
            )

            let session = LanguageModelSession(model: highTempModel)
            let response = try await session.respond(to: "Say something creative")
            #expect(!response.content.isEmpty)
        }

        @Test func maxTokensLimit() async throws {
            let session = LanguageModelSession(model: model)

            let options = GenerationOptions(maximumResponseTokens: 10)
            let response = try await session.respond(
                to: "Write a long essay about artificial intelligence",
                options: options
            )

            // Response should be limited by max tokens
            #expect(!response.content.isEmpty)
        }

        @Test func greedySamplingWithTemperature() async throws {
            let session = LanguageModelSession(model: model)
            let options = GenerationOptions(
                sampling: .greedy,
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
                temperature: 0.8,
                maximumResponseTokens: 50
            )

            // Set llama.cpp-specific custom options
            options[custom: LlamaLanguageModel.self] = .init(
                repeatPenalty: 1.2,
                repeatLastN: 128,
                frequencyPenalty: 0.1,
                presencePenalty: 0.1
            )

            let response = try await session.respond(
                to: "Tell me a short fact",
                options: options
            )
            #expect(!response.content.isEmpty)
        }

        @Test func withMirostatSampling() async throws {
            let session = LanguageModelSession(model: model)

            var options = GenerationOptions(
                temperature: 0.8,
                maximumResponseTokens: 50
            )

            // Use mirostat v2 for adaptive perplexity control
            options[custom: LlamaLanguageModel.self] = .init(
                mirostat: .v2(tau: 5.0, eta: 0.1)
            )

            let response = try await session.respond(
                to: "Tell me a short fact",
                options: options
            )
            #expect(!response.content.isEmpty)
        }

        @Test func multimodal_rejectsImageURL() async throws {
            let session = LanguageModelSession(model: model)
            let imageSegment = Transcript.ImageSegment(url: testImageURL)
            do {
                _ = try await session.respond(to: "Describe this image", image: imageSegment)
                Issue.record("Expected error when image segments are present")
            } catch let error as LlamaLanguageModelError {
                #expect(error == .unsupportedFeature)
            }
        }

        @Test func multimodal_rejectsImageData() async throws {
            let session = LanguageModelSession(model: model)
            let imageSegment = Transcript.ImageSegment(data: testImageData, mimeType: "image/png")
            do {
                _ = try await session.respond(to: "Describe this image", image: imageSegment)
                Issue.record("Expected error when image segments are present")
            } catch let error as LlamaLanguageModelError {
                #expect(error == .unsupportedFeature)
            }
        }
    }
#endif  // Llama
