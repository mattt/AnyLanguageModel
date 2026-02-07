import Testing

@testable import AnyLanguageModel

struct StructuredGenerationTests {
    private func baseTokenMaps() -> (tokenToText: [Int: String], textToTokens: [String: [Int]]) {
        let quoteToken = 0
        let commaToken = 1
        let rightBraceToken = 2
        let rightBracketToken = 3
        let colonToken = 4
        let zeroToken = 5
        let oneToken = 6
        let twoToken = 7
        let aToken = 8
        let bToken = 9
        let xToken = 10
        let yToken = 11
        let zToken = 12
        let minusToken = 13
        let minusOneToken = 14
        let eosToken = 50

        let tokenToText: [Int: String] = [
            quoteToken: "\"",
            commaToken: ",",
            rightBraceToken: "}",
            rightBracketToken: "]",
            colonToken: ":",
            zeroToken: "0",
            oneToken: "1",
            twoToken: "2",
            aToken: "a",
            bToken: "b",
            xToken: "x",
            yToken: "y",
            zToken: "z",
            minusToken: "-",
            minusOneToken: "-1",
            eosToken: "<eos>",
        ]

        let textToTokens: [String: [Int]] = [
            "\"": [quoteToken],
            ",": [commaToken],
            "}": [rightBraceToken],
            "]": [rightBracketToken],
            ":": [colonToken],
            "0": [zeroToken],
            "1": [oneToken],
            "2": [twoToken],
            "a": [aToken],
            "b": [bToken],
            "x": [xToken],
            "y": [yToken],
            "z": [zToken],
            "-": [minusToken],
            "-1": [minusOneToken],
            "ab": [aToken, bToken],
        ]

        return (tokenToText, textToTokens)
    }

    @Test func numberOutOfRangeThrows() async throws {
        let maps = baseTokenMaps()
        let numberNode = GenerationSchema.NumberNode(
            description: nil,
            minimum: 0,
            maximum: 10,
            integerOnly: true
        )
        let schema = GenerationSchema.primitive(Int.self, node: .number(numberNode))
        let eosToken = 50
        let rightBraceToken = 2
        let oneToken = 6
        let backend = MockTokenBackend(
            tokenToText: maps.tokenToText,
            textToTokens: maps.textToTokens,
            eosToken: eosToken,
            endTokens: [eosToken],
            maximumTokens: 8,
            samplingQueue: [oneToken, oneToken, rightBraceToken]
        )

        var generator = try ConstrainedJSONGenerator(backend: backend, schema: schema)
        do {
            _ = try await generator.generate()
            Issue.record("Expected number out-of-range error.")
        } catch let error as ConstrainedGenerationError {
            guard case .numberOutOfRange = error else {
                Issue.record("Unexpected error: \(error).")
                return
            }
        }
    }

    @Test func patternMismatchThrows() async throws {
        let maps = baseTokenMaps()
        let stringNode = GenerationSchema.StringNode(
            description: nil,
            pattern: "^abc$",
            enumChoices: nil
        )
        let schema = GenerationSchema.primitive(String.self, node: .string(stringNode))
        let eosToken = 50
        let quoteToken = 0
        let xToken = 10
        let yToken = 11
        let zToken = 12
        let backend = MockTokenBackend(
            tokenToText: maps.tokenToText,
            textToTokens: maps.textToTokens,
            eosToken: eosToken,
            endTokens: [eosToken],
            maximumTokens: 8,
            samplingQueue: [xToken, yToken, zToken, quoteToken]
        )

        var generator = try ConstrainedJSONGenerator(backend: backend, schema: schema)
        do {
            _ = try await generator.generate()
            Issue.record("Expected pattern mismatch error.")
        } catch let error as ConstrainedGenerationError {
            guard case .patternMismatch = error else {
                Issue.record("Unexpected error: \(error).")
                return
            }
        }
    }

    @Test func emptyStringEnumProducesEmptyValue() async throws {
        let maps = baseTokenMaps()
        let stringNode = GenerationSchema.StringNode(
            description: nil,
            pattern: nil,
            enumChoices: ["", "a"]
        )
        let schema = GenerationSchema.primitive(String.self, node: .string(stringNode))
        let eosToken = 50
        let backend = MockTokenBackend(
            tokenToText: maps.tokenToText,
            textToTokens: maps.textToTokens,
            eosToken: eosToken,
            endTokens: [eosToken],
            maximumTokens: 5
        )

        var generator = try ConstrainedJSONGenerator(backend: backend, schema: schema)
        let result = try await generator.generate()
        #expect(result == "\"\"")
    }

    @Test func prefixEnumSelectsLongerCandidateDeterministically() async throws {
        let maps = baseTokenMaps()
        let stringNode = GenerationSchema.StringNode(
            description: nil,
            pattern: nil,
            enumChoices: ["a", "ab"]
        )
        let schema = GenerationSchema.primitive(String.self, node: .string(stringNode))
        let eosToken = 50
        let backend = MockTokenBackend(
            tokenToText: maps.tokenToText,
            textToTokens: maps.textToTokens,
            eosToken: eosToken,
            endTokens: [eosToken],
            maximumTokens: 4
        )

        var generator = try ConstrainedJSONGenerator(backend: backend, schema: schema)
        let result = try await generator.generate()
        #expect(result == "\"ab\"")
    }

    @Test func eosStopsGenerationAndReturnsPartialOutput() async throws {
        let maps = baseTokenMaps()
        let stringNode = GenerationSchema.StringNode(
            description: nil,
            pattern: nil,
            enumChoices: nil
        )
        let schema = GenerationSchema.primitive(String.self, node: .string(stringNode))
        let eosToken = 50
        let aToken = 8
        let backend = MockTokenBackend(
            tokenToText: maps.tokenToText,
            textToTokens: maps.textToTokens,
            eosToken: eosToken,
            endTokens: [eosToken],
            maximumTokens: 4,
            samplingQueue: [aToken, eosToken]
        )

        var generator = try ConstrainedJSONGenerator(backend: backend, schema: schema)
        let result = try await generator.generate()
        #expect(result == "\"a")
    }

    @Test func multiTokenStructuralEncodingThrows() throws {
        var maps = baseTokenMaps()
        let eosToken = 50
        maps.textToTokens[","] = [1, 1]
        let stringNode = GenerationSchema.StringNode(
            description: nil,
            pattern: nil,
            enumChoices: nil
        )
        let schema = GenerationSchema.primitive(String.self, node: .string(stringNode))

        do {
            _ = try ConstrainedJSONGenerator(
                backend: MockTokenBackend(
                    tokenToText: maps.tokenToText,
                    textToTokens: maps.textToTokens,
                    eosToken: eosToken,
                    endTokens: [eosToken],
                    maximumTokens: 4
                ),
                schema: schema
            )
            Issue.record("Expected unsupported tokenizer error.")
        } catch let error as ConstrainedGenerationError {
            guard case .unsupportedTokenizer = error else {
                Issue.record("Unexpected error: \(error).")
                return
            }
        }
    }

    @Test func outputMatchesDecodedTokens() async throws {
        let maps = baseTokenMaps()
        let stringNode = GenerationSchema.StringNode(
            description: nil,
            pattern: nil,
            enumChoices: nil
        )
        let schema = GenerationSchema.primitive(String.self, node: .string(stringNode))
        let eosToken = 50
        let quoteToken = 0
        let aToken = 8
        let backend = MockTokenBackend(
            tokenToText: maps.tokenToText,
            textToTokens: maps.textToTokens,
            eosToken: eosToken,
            endTokens: [eosToken],
            maximumTokens: 4,
            samplingQueue: [aToken, quoteToken]
        )
        let capture = backend.capture

        var generator = try ConstrainedJSONGenerator(backend: backend, schema: schema)
        let result = try await generator.generate()
        #expect(result == capture.decodedText)
    }

    @Test func negativeIntegerWithinRange() async throws {
        let maps = baseTokenMaps()
        let numberNode = GenerationSchema.NumberNode(
            description: nil,
            minimum: -10,
            maximum: 0,
            integerOnly: true
        )
        let schema = GenerationSchema.primitive(Int.self, node: .number(numberNode))
        let eosToken = 50
        let minusOneToken = 14
        let rightBraceToken = 2
        let backend = MockTokenBackend(
            tokenToText: maps.tokenToText,
            textToTokens: maps.textToTokens,
            eosToken: eosToken,
            endTokens: [eosToken],
            maximumTokens: 6,
            samplingQueue: [minusOneToken, rightBraceToken]
        )

        var generator = try ConstrainedJSONGenerator(backend: backend, schema: schema)
        let result = try await generator.generate()
        #expect(result == "-1")
    }

    @Test func decimalOutOfRangeThrows() async throws {
        let maps = baseTokenMaps()
        let numberNode = GenerationSchema.NumberNode(
            description: nil,
            minimum: nil,
            maximum: 1,
            integerOnly: false
        )
        let schema = GenerationSchema.primitive(Double.self, node: .number(numberNode))
        let eosToken = 50
        let twoToken = 7
        let rightBraceToken = 2
        let backend = MockTokenBackend(
            tokenToText: maps.tokenToText,
            textToTokens: maps.textToTokens,
            eosToken: eosToken,
            endTokens: [eosToken],
            maximumTokens: 6,
            samplingQueue: [twoToken, rightBraceToken]
        )

        var generator = try ConstrainedJSONGenerator(backend: backend, schema: schema)
        do {
            _ = try await generator.generate()
            Issue.record("Expected number out-of-range error.")
        } catch let error as ConstrainedGenerationError {
            guard case .numberOutOfRange = error else {
                Issue.record("Unexpected error: \(error).")
                return
            }
        }
    }

    @Test func tokenBudgetExceededThrows() async throws {
        let maps = baseTokenMaps()
        let stringNode = GenerationSchema.StringNode(
            description: nil,
            pattern: nil,
            enumChoices: ["a"]
        )
        let schema = GenerationSchema.primitive(String.self, node: .string(stringNode))
        let eosToken = 50
        let backend = MockTokenBackend(
            tokenToText: maps.tokenToText,
            textToTokens: maps.textToTokens,
            eosToken: eosToken,
            endTokens: [eosToken],
            maximumTokens: 0
        )

        var generator = try ConstrainedJSONGenerator(backend: backend, schema: schema)
        do {
            _ = try await generator.generate()
            Issue.record("Expected token budget exceeded error.")
        } catch let error as ConstrainedGenerationError {
            guard case .tokenBudgetExceeded = error else {
                Issue.record("Unexpected error: \(error).")
                return
            }
        }
    }

    @Test func anyOfSingleVariantUsesOnlyChoice() async throws {
        let maps = baseTokenMaps()
        let stringNode = GenerationSchema.StringNode(
            description: nil,
            pattern: nil,
            enumChoices: ["a"]
        )
        let schema = GenerationSchema.primitive(
            String.self,
            node: .anyOf([.string(stringNode)])
        )
        let eosToken = 50
        let backend = MockTokenBackend(
            tokenToText: maps.tokenToText,
            textToTokens: maps.textToTokens,
            eosToken: eosToken,
            endTokens: [eosToken],
            maximumTokens: 4
        )

        var generator = try ConstrainedJSONGenerator(backend: backend, schema: schema)
        let result = try await generator.generate()
        #expect(result == "\"a\"")
    }

    @Test func multiTokenQuoteEncodingThrows() throws {
        var maps = baseTokenMaps()
        let eosToken = 50
        maps.textToTokens["\""] = [0, 0]
        let stringNode = GenerationSchema.StringNode(
            description: nil,
            pattern: nil,
            enumChoices: nil
        )
        let schema = GenerationSchema.primitive(String.self, node: .string(stringNode))

        do {
            _ = try ConstrainedJSONGenerator(
                backend: MockTokenBackend(
                    tokenToText: maps.tokenToText,
                    textToTokens: maps.textToTokens,
                    eosToken: eosToken,
                    endTokens: [eosToken],
                    maximumTokens: 4
                ),
                schema: schema
            )
            Issue.record("Expected unsupported tokenizer error.")
        } catch let error as ConstrainedGenerationError {
            guard case .unsupportedTokenizer = error else {
                Issue.record("Unexpected error: \(error).")
                return
            }
        }
    }

    @Test func invalidArrayBoundsThrows() async throws {
        let maps = baseTokenMaps()
        let arrayNode = GenerationSchema.ArrayNode(
            description: nil,
            items: .string(.init()),
            minItems: 3,
            maxItems: 1
        )
        let schema = GenerationSchema.primitive([String].self, node: .array(arrayNode))
        let eosToken = 50
        let backend = MockTokenBackend(
            tokenToText: maps.tokenToText,
            textToTokens: maps.textToTokens,
            eosToken: eosToken,
            endTokens: [eosToken],
            maximumTokens: 6
        )

        var generator = try ConstrainedJSONGenerator(backend: backend, schema: schema)
        do {
            _ = try await generator.generate()
            Issue.record("Expected invalid array bounds error.")
        } catch let error as ConstrainedGenerationError {
            guard case .invalidArrayBounds = error else {
                Issue.record("Unexpected error: \(error).")
                return
            }
        }
    }

    @Test func arrayCountIsDeterministic() async throws {
        let maps = baseTokenMaps()
        let arrayNode = GenerationSchema.ArrayNode(
            description: nil,
            items: .string(.init(enumChoices: ["a"])),
            minItems: 1,
            maxItems: 3
        )
        let schema = GenerationSchema.primitive([String].self, node: .array(arrayNode))
        let eosToken = 50
        let backend = MockTokenBackend(
            tokenToText: maps.tokenToText,
            textToTokens: maps.textToTokens,
            eosToken: eosToken,
            endTokens: [eosToken],
            maximumTokens: 17
        )

        var generator = try ConstrainedJSONGenerator(backend: backend, schema: schema)
        let result = try await generator.generate()
        #expect(result == "[\"a\",\"a\",\"a\"]")
    }
}
