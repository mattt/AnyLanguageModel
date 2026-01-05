import Foundation
import Testing

@testable import AnyLanguageModel

#if CoreML
    import Hub
    import CoreML

    private let shouldRunCoreMLTests: Bool = {
        // Enable when explicitly requested via environment variable
        if ProcessInfo.processInfo.environment["ENABLE_COREML_TESTS"] != nil {
            return true
        }

        // Skip in CI environments
        if ProcessInfo.processInfo.environment["CI"] != nil {
            return false
        }

        return true
    }()

    @Suite("CoreMLLanguageModel", .enabled(if: shouldRunCoreMLTests))
    struct CoreMLLanguageModelTests {
        let modelId = "apple/mistral-coreml"
        let modelPackageName = "StatefulMistral7BInstructInt4.mlpackage"

        @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
        func getModel() async throws -> CoreMLLanguageModel {
            let hasToken = ProcessInfo.processInfo.environment["HF_TOKEN"] != nil
            let hubApi = HubApi(useOfflineMode: !hasToken)
            let repoURL = try await hubApi.snapshot(
                from: Hub.Repo(id: modelId, type: .models),
                matching: "*Int4.mlpackage/**"
            ) { progress in
                print("Download progress: \(Int(progress.fractionCompleted * 100))%")
            }

            let modelURL = repoURL.appending(component: modelPackageName)
            return try await CoreMLLanguageModel(url: modelURL)
        }

        @Test @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
        func basicResponse() async throws {
            let model = try await getModel()
            let session = LanguageModelSession(model: model)

            let response = try await session.respond(to: "Say hello")
            #expect(!response.content.isEmpty)
        }

        @Test @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
        func withGenerationOptions() async throws {
            let model = try await getModel()
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

        @Test @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
        func withSamplingStrategies() async throws {
            let model = try await getModel()
            let session = LanguageModelSession(model: model)

            // Test greedy sampling
            let greedyOptions = GenerationOptions(sampling: .greedy)
            let greedyResponse = try await session.respond(
                to: "Complete this sentence: The sky is",
                options: greedyOptions
            )
            #expect(!greedyResponse.content.isEmpty)

            // Test top-k sampling
            let topKOptions = GenerationOptions(sampling: .random(top: 10))
            let topKResponse = try await session.respond(
                to: "Complete this sentence: The sky is",
                options: topKOptions
            )
            #expect(!topKResponse.content.isEmpty)

            // Test nucleus sampling
            let nucleusOptions = GenerationOptions(sampling: .random(probabilityThreshold: 0.9))
            let nucleusResponse = try await session.respond(
                to: "Complete this sentence: The sky is",
                options: nucleusOptions
            )
            #expect(!nucleusResponse.content.isEmpty)
        }

        @Test @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
        func temperatureVariations() async throws {
            let model = try await getModel()
            let session = LanguageModelSession(model: model)

            // Test low temperature (more deterministic)
            let lowTempOptions = GenerationOptions(temperature: 0.1)
            let lowTempResponse = try await session.respond(
                to: "Write a short story about a cat",
                options: lowTempOptions
            )
            #expect(!lowTempResponse.content.isEmpty)

            // Test high temperature (more creative)
            let highTempOptions = GenerationOptions(temperature: 0.9)
            let highTempResponse = try await session.respond(
                to: "Write a short story about a cat",
                options: highTempOptions
            )
            #expect(!highTempResponse.content.isEmpty)
        }

        @Test @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
        func maxTokensLimit() async throws {
            let model = try await getModel()
            let session = LanguageModelSession(model: model)

            let options = GenerationOptions(maximumResponseTokens: 5)
            let response = try await session.respond(
                to: "Write a long story about space exploration",
                options: options
            )
            #expect(!response.content.isEmpty)
            // Note: We can't easily test token count without access to the tokenizer
            // but we can verify the response is not empty
        }

        @Test @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
        func multimodal_rejectsImageURL() async throws {
            let model = try await getModel()
            let session = LanguageModelSession(model: model)
            do {
                _ = try await session.respond(
                    to: "Describe this image",
                    image: .init(url: testImageURL)
                )
                Issue.record("Expected error when image segments are present")
            } catch {
                // CoreMLUnsupportedFeatureError is a private struct, so we just check that an error is thrown
                #expect(Bool(true))
            }
        }

        @Test @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
        func multimodal_rejectsImageData() async throws {
            let model = try await getModel()
            let session = LanguageModelSession(model: model)
            do {
                _ = try await session.respond(
                    to: "Describe this image",
                    image: .init(data: testImageData, mimeType: "image/jpeg")
                )
                Issue.record("Expected error when image segments are present")
            } catch {
                // CoreMLUnsupportedFeatureError is a private struct, so we just check that an error is thrown
                #expect(Bool(true))
            }
        }
    }
#endif  // CoreML
