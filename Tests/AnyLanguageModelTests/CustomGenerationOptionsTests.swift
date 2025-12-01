import Foundation
import Testing

@testable import AnyLanguageModel

@Suite("CustomGenerationOptions")
struct CustomGenerationOptionsTests {

    // MARK: - Protocol Conformance

    @Test func neverConformsToCustomGenerationOptions() {
        // Never should conform to CustomGenerationOptions (used as default)
        let _: any CustomGenerationOptions.Type = Never.self
    }

    // MARK: - Subscript Access

    @Test func subscriptGetReturnsNilWhenNotSet() {
        let options = GenerationOptions()
        let customOptions = options[custom: OpenAILanguageModel.self]
        #expect(customOptions == nil)
    }

    @Test func subscriptSetAndGet() {
        var options = GenerationOptions()
        let customOptions = OpenAILanguageModel.CustomGenerationOptions(
            extraBody: ["test": .string("value")]
        )

        options[custom: OpenAILanguageModel.self] = customOptions

        let retrieved = options[custom: OpenAILanguageModel.self]
        #expect(retrieved != nil)
        #expect(retrieved?.extraBody?["test"] == .string("value"))
    }

    @Test func subscriptSetToNilRemovesValue() {
        var options = GenerationOptions()
        options[custom: OpenAILanguageModel.self] = .init(extraBody: ["key": .bool(true)])

        #expect(options[custom: OpenAILanguageModel.self] != nil)

        options[custom: OpenAILanguageModel.self] = nil

        #expect(options[custom: OpenAILanguageModel.self] == nil)
    }

    @Test func subscriptIsolatesModelTypes() {
        var options = GenerationOptions()

        // Set custom options for OpenAI
        options[custom: OpenAILanguageModel.self] = .init(
            extraBody: ["openai_key": .string("openai_value")]
        )

        // MockLanguageModel uses Never as CustomGenerationOptions (default)
        // So we can't set custom options for it - this is by design
        let mockOptions: Never? = options[custom: MockLanguageModel.self]
        #expect(mockOptions == nil)

        // OpenAI options should still be accessible
        let openaiOptions = options[custom: OpenAILanguageModel.self]
        #expect(openaiOptions?.extraBody?["openai_key"] == .string("openai_value"))
    }

    // MARK: - Equality

    @Test func equalityWithNoCustomOptions() {
        let options1 = GenerationOptions(temperature: 0.7)
        let options2 = GenerationOptions(temperature: 0.7)

        #expect(options1 == options2)
    }

    @Test func equalityWithSameCustomOptions() {
        var options1 = GenerationOptions(temperature: 0.7)
        var options2 = GenerationOptions(temperature: 0.7)

        options1[custom: OpenAILanguageModel.self] = .init(extraBody: ["key": .bool(true)])
        options2[custom: OpenAILanguageModel.self] = .init(extraBody: ["key": .bool(true)])

        #expect(options1 == options2)
    }

    @Test func inequalityWithDifferentCustomOptions() {
        var options1 = GenerationOptions(temperature: 0.7)
        var options2 = GenerationOptions(temperature: 0.7)

        options1[custom: OpenAILanguageModel.self] = .init(extraBody: ["key": .bool(true)])
        options2[custom: OpenAILanguageModel.self] = .init(extraBody: ["key": .bool(false)])

        #expect(options1 != options2)
    }

    @Test func inequalityWhenOnlyOneHasCustomOptions() {
        var options1 = GenerationOptions(temperature: 0.7)
        let options2 = GenerationOptions(temperature: 0.7)

        options1[custom: OpenAILanguageModel.self] = .init(extraBody: ["key": .bool(true)])

        #expect(options1 != options2)
    }

    // MARK: - Encoding

    @Test func encodingWithCustomOptions() throws {
        var options = GenerationOptions(temperature: 0.8)
        options[custom: OpenAILanguageModel.self] = .init(
            extraBody: ["reasoning": .object(["enabled": .bool(true)])]
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
        let data = try encoder.encode(options)
        let json = String(data: data, encoding: .utf8)!

        // Verify the JSON contains the temperature
        #expect(json.contains("\"temperature\""))
        #expect(json.contains("0.8"))

        // Verify custom options type name is in the output
        #expect(json.contains("OpenAILanguageModel"))
        #expect(json.contains("CustomGenerationOptions"))
    }

    @Test func decodingLosesCustomOptions() throws {
        var options = GenerationOptions(temperature: 0.8)
        options[custom: OpenAILanguageModel.self] = .init(
            extraBody: ["key": .string("value")]
        )

        let encoder = JSONEncoder()
        let data = try encoder.encode(options)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(GenerationOptions.self, from: data)

        // Standard options should be preserved
        #expect(decoded.temperature == 0.8)

        // Custom options are lost on round-trip (documented behavior)
        #expect(decoded[custom: OpenAILanguageModel.self] == nil)
    }
}

@Suite("OpenAI CustomGenerationOptions")
struct OpenAICustomOptionsTests {
    @Test func initialization() {
        let options = OpenAILanguageModel.CustomGenerationOptions(
            extraBody: [
                "reasoning": .object(["enabled": .bool(true)]),
                "custom_param": .string("value"),
            ]
        )

        #expect(options.extraBody?.count == 2)
        #expect(options.extraBody?["reasoning"] == .object(["enabled": .bool(true)]))
    }

    @Test func equality() {
        let options1 = OpenAILanguageModel.CustomGenerationOptions(
            extraBody: ["key": .int(42)]
        )
        let options2 = OpenAILanguageModel.CustomGenerationOptions(
            extraBody: ["key": .int(42)]
        )

        #expect(options1 == options2)
    }

    @Test func hashable() {
        let options: OpenAILanguageModel.CustomGenerationOptions = OpenAILanguageModel.CustomGenerationOptions(
            extraBody: ["key": .bool(true)]
        )

        var set = Set<OpenAILanguageModel.CustomGenerationOptions>()
        set.insert(options)
        #expect(set.contains(options))
    }

    @Test func codable() throws {
        let options = OpenAILanguageModel.CustomGenerationOptions(
            extraBody: ["key": .string("value")]
        )

        let data = try JSONEncoder().encode(options)
        let decoded = try JSONDecoder().decode(
            OpenAILanguageModel.CustomGenerationOptions.self,
            from: data
        )

        #expect(decoded == options)
    }

    @Test func nilExtraBody() {
        let options = OpenAILanguageModel.CustomGenerationOptions()
        #expect(options.extraBody == nil)
    }
}

#if Llama
    @Suite("Llama CustomGenerationOptions")
    struct LlamaCustomOptionsTests {
        @Test func initialization() {
            let options = LlamaLanguageModel.CustomGenerationOptions(
                repeatPenalty: 1.2,
                repeatLastN: 128,
                frequencyPenalty: 0.1,
                presencePenalty: 0.1
            )

            #expect(options.repeatPenalty == 1.2)
            #expect(options.repeatLastN == 128)
            #expect(options.frequencyPenalty == 0.1)
            #expect(options.presencePenalty == 0.1)
        }

        @Test func mirostatModes() {
            let v1 = LlamaLanguageModel.CustomGenerationOptions(
                mirostat: .v1(tau: 5.0, eta: 0.1)
            )
            let v2 = LlamaLanguageModel.CustomGenerationOptions(
                mirostat: .v2(tau: 5.0, eta: 0.1)
            )

            #expect(v1.mirostat != nil)
            #expect(v2.mirostat != nil)
            #expect(v1 != v2)
        }

        @Test func equality() {
            let options1 = LlamaLanguageModel.CustomGenerationOptions(
                repeatPenalty: 1.1,
                repeatLastN: 64
            )
            let options2 = LlamaLanguageModel.CustomGenerationOptions(
                repeatPenalty: 1.1,
                repeatLastN: 64
            )

            #expect(options1 == options2)
        }

        @Test func codable() throws {
            let options = LlamaLanguageModel.CustomGenerationOptions(
                repeatPenalty: 1.2,
                repeatLastN: 128,
                mirostat: .v2(tau: 5.0, eta: 0.1)
            )

            let data = try JSONEncoder().encode(options)
            let decoded = try JSONDecoder().decode(
                LlamaLanguageModel.CustomGenerationOptions.self,
                from: data
            )

            #expect(decoded == options)
        }

        @Test func integrationWithGenerationOptions() {
            var options = GenerationOptions(temperature: 0.8)
            options[custom: LlamaLanguageModel.self] = .init(
                repeatPenalty: 1.2,
                repeatLastN: 128
            )

            let retrieved = options[custom: LlamaLanguageModel.self]
            #expect(retrieved?.repeatPenalty == 1.2)
            #expect(retrieved?.repeatLastN == 128)
        }
    }
#endif
