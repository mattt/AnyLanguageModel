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

@Suite("Generable Macro")
struct GenerableMacroTests {
    @Test func multilineGuideDescription() async throws {
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

    @Test func guideDescriptionWithSpecialCharacters() async throws {
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

    @Test func guideDescriptionWithNewlines() async throws {
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
}
