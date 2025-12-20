import Foundation
import Testing

@testable import AnyLanguageModel

@Generable
enum Priority: Equatable {
    case low
    case medium
    case high
}

@Generable
struct SimpleString: Equatable {
    @Guide(description: "A greeting message")
    var message: String
}

@Generable
struct SimpleInt: Equatable {
    @Guide(description: "A count value", .minimum(0))
    var count: Int
}

@Generable
struct SimpleBool: Equatable {
    @Guide(description: "A boolean flag")
    var value: Bool
}

@Generable
struct SimpleDouble: Equatable {
    @Guide(description: "A temperature value")
    var temperature: Double
}

@Generable
struct OptionalFields: Equatable {
    @Guide(description: "A required name")
    var name: String

    @Guide(description: "An optional nickname")
    var nickname: String?
}

@Generable
struct BasicStruct: Equatable {
    @Guide(description: "Person's name")
    var name: String

    @Guide(description: "Person's age", .minimum(0))
    var age: Int

    @Guide(description: "Is the person active")
    var isActive: Bool

    @Guide(description: "Score value")
    var score: Double
}

@Generable
struct Address: Equatable {
    @Guide(description: "Street name")
    var street: String

    @Guide(description: "City name")
    var city: String

    @Guide(description: "Postal code")
    var postalCode: String
}

@Generable
struct ReusedNestedStruct: Equatable {
    @Guide(description: "Some text")
    var text: String
}

@Generable
struct ContainerWithDuplicateNestedType: Equatable {
    var first: ReusedNestedStruct
    var second: ReusedNestedStruct
}

@Generable
struct Person: Equatable {
    @Guide(description: "Person's name")
    var name: String

    @Guide(description: "Person's age")
    var age: Int

    var address: Address
}

@Generable
struct TaskItem: Equatable {
    @Guide(description: "Task title")
    var title: String

    var priority: Priority

    @Guide(description: "Is completed")
    var isCompleted: Bool
}

@Generable
struct SimpleArray: Equatable {
    @Guide(description: "A list of color names")
    var colors: [String]
}

@Generable
struct MultiChoiceQuestion: Equatable {
    @Guide(description: "The quiz question")
    var text: String

    @Guide(.count(4))
    var choices: [String]

    var answer: String

    @Guide(description: "A brief explanation of why the answer is correct")
    var explanation: String
}

private struct SupportedModel: Sendable {
    let name: String
    let model: any LanguageModel

    static var all: [SupportedModel] {
        var models: [SupportedModel] = []

        #if canImport(FoundationModels)
            if #available(macOS 26.0, *) {
                if SystemLanguageModel.default.isAvailable {
                    models.append(SupportedModel(name: "SystemLanguageModel", model: SystemLanguageModel.default))
                }
            }
        #endif

        #if Llama
            if let modelPath = ProcessInfo.processInfo.environment["LLAMA_MODEL_PATH"] {
                models.append(SupportedModel(name: "LlamaLanguageModel", model: LlamaLanguageModel(modelPath: modelPath)))
            }
        #endif

        #if MLX
            let shouldRunMLX = ProcessInfo.processInfo.environment["ENABLE_MLX_TESTS"] != nil
                || (ProcessInfo.processInfo.environment["CI"] == nil
                    && ProcessInfo.processInfo.environment["HF_TOKEN"] != nil
                    && ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil)
            if shouldRunMLX {
                models.append(
                    SupportedModel(
                        name: "MLXLanguageModel",
                        model: MLXLanguageModel(modelId: "mlx-community/Qwen3-0.6B-4bit")
                    )
                )
            }
        #endif

        return models
    }
}

private let supportedModels = SupportedModel.all

private func isGenerationTestsEnabled() -> Bool {
    !supportedModels.isEmpty
}

@Test("GenerationSchema merges duplicate defs for the same type")
func generationSchemaMergesDuplicateDefsForSameType() {
    let schema = ContainerWithDuplicateNestedType.generationSchema

    let nestedTypeName = String(reflecting: ReusedNestedStruct.self)
    #expect(schema.defs[nestedTypeName] != nil)
}

private func testAllModels(_ test: (SupportedModel) async throws -> Void) async {
    var failures: [(name: String, error: any Error)] = []

    for model in supportedModels {
        do {
            try await test(model)
        } catch {
            failures.append((model.name, error))
        }
    }

    for failure in failures {
        Issue.record("[\(failure.name)] \(failure.error)")
    }
}

private func logGenerated<T: Generable>(_ content: T, model: String) {
    let json = content.generatedContent.jsonString
    if let data = json.data(using: .utf8),
       let object = try? JSONSerialization.jsonObject(with: data),
       let prettyData = try? JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys]),
       let prettyJSON = String(data: prettyData, encoding: .utf8)
    {
        print("\n[\(model)]\n\(prettyJSON)\n")
    } else {
        print("\n[\(model)]\n\(json)\n")
    }
}

@Suite("Structured Generation", .serialized, .enabled(if: isGenerationTestsEnabled()))
struct StructuredGenerationTests {
    @Test("Generate SimpleString with all supported models")
    func generateSimpleString() async {
        await testAllModels { model in
            let session = LanguageModelSession(
                model: model.model,
                instructions: "You are a helpful assistant that generates structured data."
            )

            let response = try await session.respond(
                to: "Generate a greeting message that says hello",
                generating: SimpleString.self
            )

            logGenerated(response.content, model: model.name)
            #expect(!response.content.message.isEmpty, "[\(model.name)] message should not be empty")
        }
    }

    @Test("Generate SimpleInt with all supported models")
    func generateSimpleInt() async {
        await testAllModels { model in
            let session = LanguageModelSession(
                model: model.model,
                instructions: "You are a helpful assistant that generates structured data."
            )

            let response = try await session.respond(
                to: "Generate a count value of 42",
                generating: SimpleInt.self
            )

            logGenerated(response.content, model: model.name)
            #expect(response.content.count >= 0, "[\(model.name)] count should be non-negative")
        }
    }

    @Test("Generate SimpleDouble with all supported models")
    func generateSimpleDouble() async {
        await testAllModels { model in
            let session = LanguageModelSession(
                model: model.model,
                instructions: "You are a helpful assistant that generates structured data."
            )

            let response = try await session.respond(
                to: "Generate a temperature value of 72.5 degrees",
                generating: SimpleDouble.self
            )

            logGenerated(response.content, model: model.name)
            #expect(!response.content.temperature.isNaN, "[\(model.name)] temperature should be a valid number")
        }
    }

    @Test("Generate SimpleBool with all supported models")
    func generateSimpleBool() async {
        await testAllModels { model in
            let session = LanguageModelSession(
                model: model.model,
                instructions: "You are a helpful assistant that generates structured data."
            )

            let response = try await session.respond(
                to: "Generate a boolean value: true",
                generating: SimpleBool.self
            )

            logGenerated(response.content, model: model.name)
            let jsonData = response.rawContent.jsonString.data(using: .utf8)
            #expect(jsonData != nil, "[\(model.name)] rawContent should be valid UTF-8 JSON")
            if let jsonData {
                let json = try JSONSerialization.jsonObject(with: jsonData)
                let dictionary = json as? [String: Any]
                let boolValue = dictionary?["value"] as? Bool
                #expect(boolValue != nil, "[\(model.name)] value should be encoded as a JSON boolean")
            }
        }
    }

    @Test("Generate OptionalFields with all supported models")
    func generateOptionalFields() async {
        await testAllModels { model in
            let session = LanguageModelSession(
                model: model.model,
                instructions: "You are a helpful assistant that generates structured data."
            )

            let response = try await session.respond(
                to: "Generate a person named Alex with nickname 'Lex'. Nickname may be omitted if unsure.",
                generating: OptionalFields.self
            )

            logGenerated(response.content, model: model.name)
            #expect(!response.content.name.isEmpty, "[\(model.name)] name should not be empty")
            if let nickname = response.content.nickname {
                #expect(!nickname.isEmpty, "[\(model.name)] nickname should not be empty when present")
            }
        }
    }

    @Test("Generate Priority enum with all supported models")
    func generatePriorityEnum() async {
        await testAllModels { model in
            let session = LanguageModelSession(
                model: model.model,
                instructions: "You are a helpful assistant that generates structured data."
            )

            let response = try await session.respond(
                to: "Generate a high priority value",
                generating: Priority.self
            )

            logGenerated(response.content, model: model.name)
            #expect(
                [Priority.low, Priority.medium, Priority.high].contains(response.content),
                "[\(model.name)] should generate valid priority"
            )
        }
    }

    @Test("Generate BasicStruct with all supported models")
    func generateBasicStruct() async {
        await testAllModels { model in
            let session = LanguageModelSession(
                model: model.model,
                instructions: "You are a helpful assistant that generates structured data."
            )

            let response = try await session.respond(
                to: "Generate a person with name Alice, age 30, active status true, and score 95.5",
                generating: BasicStruct.self
            )

            logGenerated(response.content, model: model.name)
            #expect(!response.content.name.isEmpty, "[\(model.name)] name should not be empty")
            #expect(response.content.age >= 0, "[\(model.name)] age should be non-negative")
        }
    }

    @Test("Generate nested struct (Person with Address) with all supported models")
    func generateNestedStruct() async {
        await testAllModels { model in
            let session = LanguageModelSession(
                model: model.model,
                instructions: "You are a helpful assistant that generates structured data."
            )

            let response = try await session.respond(
                to: "Generate a person named John, age 25, living at 123 Main St, Springfield, 12345",
                generating: Person.self
            )

            logGenerated(response.content, model: model.name)
            #expect(!response.content.name.isEmpty, "[\(model.name)] name should not be empty")
            #expect(response.content.age >= 0, "[\(model.name)] age should be non-negative")
            #expect(!response.content.address.street.isEmpty, "[\(model.name)] street should not be empty")
            #expect(!response.content.address.city.isEmpty, "[\(model.name)] city should not be empty")
        }
    }

    @Test("Generate struct with enum (TaskItem) with all supported models")
    func generateStructWithEnum() async {
        await testAllModels { model in
            let session = LanguageModelSession(
                model: model.model,
                instructions: "You are a helpful assistant that generates structured data."
            )

            let response = try await session.respond(
                to: "Generate a task titled 'Complete project' with high priority, not completed",
                generating: TaskItem.self
            )

            logGenerated(response.content, model: model.name)
            #expect(!response.content.title.isEmpty, "[\(model.name)] title should not be empty")
            #expect(
                [Priority.low, Priority.medium, Priority.high].contains(response.content.priority),
                "[\(model.name)] should have valid priority"
            )
        }
    }

    @Test("Generate simple array with all supported models")
    func generateSimpleArray() async {
        await testAllModels { model in
            let session = LanguageModelSession(
                model: model.model,
                instructions: "You are a helpful assistant that generates structured data."
            )

            let response = try await session.respond(
                to: "Generate a list of 3 color names: red, green, blue",
                generating: SimpleArray.self
            )

            logGenerated(response.content, model: model.name)
            #expect(!response.content.colors.isEmpty, "[\(model.name)] colors should not be empty")
        }
    }

    @Test("Generate struct with array (MultiChoiceQuestion) with all supported models")
    func generateStructWithArray() async {
        await testAllModels { model in
            let session = LanguageModelSession(
                model: model.model,
                instructions: "You are a helpful assistant that generates structured data."
            )

            let response = try await session.respond(
                to: """
                    Generate a quiz question:
                    - Question: What is the capital of France?
                    - Choices: London, Paris, Berlin, Madrid
                    - Answer: Paris
                    - Explanation: Paris is the capital city of France
                    """,
                generating: MultiChoiceQuestion.self
            )

            logGenerated(response.content, model: model.name)
            #expect(!response.content.text.isEmpty, "[\(model.name)] question text should not be empty")
            #expect(response.content.choices.count == 4, "[\(model.name)] should have exactly 4 choices")
            #expect(!response.content.answer.isEmpty, "[\(model.name)] answer should not be empty")
        }
    }
}
