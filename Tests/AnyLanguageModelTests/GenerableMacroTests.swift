import Testing
import AnyLanguageModel
import Foundation

@Generable
private struct TestStructWithMultilineDescription {
    @Guide(
        description: """
            This is a multi-line description.
            It spans multiple lines.
            """
    )
    var field: String
}

@Generable
private struct TestStructWithSpecialCharacters {
    @Guide(description: "A description with \"quotes\" and backslashes \\")
    var field: String
}

@Generable
private struct TestStructWithNewlines {
    @Guide(description: "Line 1\nLine 2\nLine 3")
    var field: String
}

@Generable
struct TestArguments {
    @Guide(description: "A name field")
    var name: String

    @Guide(description: "An age field")
    var age: Int
}

@Generable
private struct ArrayItem {
    @Guide(description: "A name")
    var name: String
}

@Generable
private struct ArrayContainer {
    @Guide(description: "Items", .count(2))
    var items: [ArrayItem]
}

@Suite("Generable Macro")
struct GenerableMacroTests {
    @Test("@Guide description with multiline string")
    func multilineGuideDescription() async throws {
        let schema = TestStructWithMultilineDescription.generationSchema
        let encoder = JSONEncoder()
        let jsonData = try encoder.encode(schema)

        // Verify that the schema can be encoded without errors (no unterminated strings)
        #expect(jsonData.count > 0)

        // Verify it can be decoded back
        let decoder = JSONDecoder()
        let decodedSchema = try decoder.decode(GenerationSchema.self, from: jsonData)
        #expect(decodedSchema.debugDescription.contains("object"))
    }

    @Test("@Guide description with special characters")
    func guideDescriptionWithSpecialCharacters() async throws {
        let schema = TestStructWithSpecialCharacters.generationSchema
        let encoder = JSONEncoder()
        let jsonData = try encoder.encode(schema)
        let jsonString = String(data: jsonData, encoding: .utf8)!

        // Verify the special characters are escaped
        #expect(jsonString.contains(#"\\\"quotes\\\""#))
        #expect(jsonString.contains(#"backslashes \\\\"#))

        // Verify roundtrip encoding/decoding works
        let decoder = JSONDecoder()
        let decodedSchema = try decoder.decode(GenerationSchema.self, from: jsonData)
        #expect(decodedSchema.debugDescription.contains("object"))
    }

    @Test("@Guide description with newlines")
    func guideDescriptionWithNewlines() async throws {
        let schema = TestStructWithNewlines.generationSchema
        let encoder = JSONEncoder()
        let jsonData = try encoder.encode(schema)

        // Verify that the schema can be encoded without errors
        #expect(jsonData.count > 0)

        // Verify roundtrip encoding/decoding works
        let decoder = JSONDecoder()
        let decodedSchema = try decoder.decode(GenerationSchema.self, from: jsonData)
        #expect(decodedSchema.debugDescription.contains("object"))
    }

    @MainActor
    @Generable
    struct MainActorIsolatedStruct {
        @Guide(description: "A test field")
        var field: String
    }

    @MainActor
    @Test("@MainActor isolation")
    func mainActorIsolation() async throws {
        let generatedContent = GeneratedContent(properties: [
            "field": "test value"
        ])
        let instance = try MainActorIsolatedStruct(generatedContent)
        #expect(instance.field == "test value")

        let convertedBack = instance.generatedContent
        let decoded = try MainActorIsolatedStruct(convertedBack)
        #expect(decoded.field == "test value")

        let schema = MainActorIsolatedStruct.generationSchema
        #expect(schema.debugDescription.contains("MainActorIsolatedStruct"))

        let partiallyGenerated = instance.asPartiallyGenerated()
        #expect(partiallyGenerated.field == "test value")
    }

    @Test("Memberwise initializer")
    func memberwiseInit() throws {
        // This is the natural Swift way to create instances
        let args = TestArguments(name: "Alice", age: 30)

        #expect(args.name == "Alice")
        #expect(args.age == 30)

        // The generatedContent should also be properly populated
        let content = args.generatedContent
        #expect(content.jsonString.contains("Alice"))
        #expect(content.jsonString.contains("30"))
    }

    @Test("Create instance from GeneratedContent")
    func fromGeneratedContent() throws {
        let generationID = GenerationID()
        let content = GeneratedContent(
            properties: [
                "name": GeneratedContent("Bob"),
                "age": GeneratedContent(kind: .number(25)),
            ],
            id: generationID
        )

        let args = try TestArguments(content)
        #expect(args.name == "Bob")
        #expect(args.age == 25)
        #expect(args.asPartiallyGenerated().id == generationID)
    }

    @Test("Array properties use partially generated element types")
    func arrayPropertyPartialTypes() throws {
        let content = GeneratedContent(
            properties: [
                "items": GeneratedContent(
                    kind: .array([
                        GeneratedContent(properties: ["name": "Alpha"]),
                        GeneratedContent(properties: ["name": "Beta"]),
                    ])
                ),
            ]
        )

        let container = try ArrayContainer(content)
        let partial = container.asPartiallyGenerated()
        #expect(partial.items?.count == 2)
        #expect(partial.items?.first?.name == "Alpha")
    }
}

// MARK: - #Playground Usage

// The `#Playground` macro doesn't see the memberwise initializer
// that `@Generable` expands. This is a limitation of how macros compose:
// one macro's expansion isn't visible within another macro's body.
//
// The following code demonstrates workarounds for this limitation.

#if canImport(Playgrounds)
    import Playgrounds

    // Use the `GeneratedContent` initializer explicitly.
    #Playground {
        let content = GeneratedContent(properties: [
            "name": "Alice",
            "age": 30,
        ])
        let _ = try TestArguments(content)
    }

    // Define a factory method as an alternative to the memberwise initializer.
    extension TestArguments {
        static func create(name: String, age: Int) -> TestArguments {
            try! TestArguments(
                GeneratedContent(properties: [
                    "name": name,
                    "age": age,
                ])
            )
        }
    }

    #Playground {
        let _ = TestArguments.create(name: "Bob", age: 42)
    }
#endif  // canImport(Playgrounds)
