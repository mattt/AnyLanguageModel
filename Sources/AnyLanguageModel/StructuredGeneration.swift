import Foundation

// MARK: - Token Backend

/// A token-level backend used by ``ConstrainedJSONGenerator``.
///
/// Implementations provide tokenization, sampling, and decoding state so the
/// generator can constrain output to valid JSON for a schema.
package protocol TokenBackend {
    func tokenize(_ text: String) throws -> [Int]
    func tokenText(_ token: Int) -> String?
    func isSpecialToken(_ token: Int) -> Bool
    mutating func decode(_ token: Int) throws
    mutating func sample(from allowedTokens: Set<Int>) throws -> Int

    var eosToken: Int { get }
    var endTokens: Set<Int> { get }
    var vocabSize: Int { get }
    var remainingTokens: Int { get set }
    var totalTokenBudget: Int { get }
}

// MARK: - JSON Generator

/// Generates JSON that conforms to a schema using constrained token sampling.
package struct ConstrainedJSONGenerator<Backend: TokenBackend> {
    private var backend: Backend
    private let schema: GenerationSchema
    private var emittedText = ""

    private let quoteToken: Int

    private let stringTerminators: Set<Int>
    private let stringInitialAllowedTokens: Set<Int>
    private let stringContinuationAllowedTokens: Set<Int>

    private let basicTerminators: Set<Int>
    private let integerTerminators: Set<Int>
    private let doubleTerminators: Set<Int>

    /// Creates a constrained JSON generator.
    ///
    /// - Parameters:
    ///   - backend: A backend that provides tokenization and sampling.
    ///   - schema: The generation schema to satisfy.
    /// - Throws: ``ConstrainedGenerationError`` when required tokens cannot be tokenized.
    package init(backend: Backend, schema: GenerationSchema) throws {
        self.backend = backend
        self.schema = schema

        let quoteToken = try Self.singleToken(for: "\"", backend: backend)
        self.quoteToken = quoteToken

        self.stringTerminators = backend.endTokens.union([quoteToken])

        var structuralTerminators = backend.endTokens
        for structuralText in [",", "}", "]", ":"] {
            let token = try Self.singleToken(for: structuralText, backend: backend)
            structuralTerminators.insert(token)
        }
        self.basicTerminators = structuralTerminators
        self.integerTerminators = Self.buildValidIntegerTokens(backend: backend).union(structuralTerminators)
        self.doubleTerminators = Self.buildValidDecimalTokens(backend: backend).union(structuralTerminators)

        let stringContentTokens = Self.buildValidStringTokens(backend: backend)
        self.stringInitialAllowedTokens = stringContentTokens
        self.stringContinuationAllowedTokens = stringContentTokens.union(stringTerminators)
    }

    /// Generates a JSON string that conforms to the schema.
    ///
    /// - Returns: A JSON string that satisfies the schema.
    /// - Throws: ``ConstrainedGenerationError`` if generation fails.
    package mutating func generate() throws -> String {
        do {
            return try generateNode(schema.root)
        } catch let error as ConstrainedGenerationError {
            if case .earlyTermination(let partial) = error {
                return partial
            }
            throw error
        }
    }

    private static func singleToken(for text: String, backend: Backend) throws -> Int {
        let tokens = try backend.tokenize(text)
        guard tokens.count == 1, let token = tokens.first else {
            throw ConstrainedGenerationError.unsupportedTokenizer("Expected single-token encoding for '\(text)'")
        }
        return token
    }

    private static func buildValidStringTokens(backend: Backend) -> Set<Int> {
        let allowedWhitespace: Set<Character> = [" ", "\t", "\n"]
        var allowed = Set<Int>()
        allowed.reserveCapacity(backend.vocabSize / 4)

        for token in 0 ..< backend.vocabSize {
            if backend.endTokens.contains(token) { continue }
            if backend.isSpecialToken(token) { continue }
            guard let text = backend.tokenText(token), !text.isEmpty else { continue }
            guard text.allSatisfy({ $0.isValidJSONStringCharacter }) else { continue }

            if text.allSatisfy({ $0.isWhitespace }) {
                if text.count == 1, let char = text.first, allowedWhitespace.contains(char) {
                    allowed.insert(token)
                }
            } else {
                allowed.insert(token)
            }
        }
        return allowed
    }

    private static func buildValidIntegerTokens(backend: Backend) -> Set<Int> {
        var allowed = Set<Int>()
        for token in 0 ..< backend.vocabSize {
            guard let text = backend.tokenText(token), !text.isEmpty else { continue }
            if text.allSatisfy({ $0.isNumber || $0 == "-" }),
                text.contains(where: { $0.isNumber })
            {
                allowed.insert(token)
            }
        }
        return allowed
    }

    private static func buildValidDecimalTokens(backend: Backend) -> Set<Int> {
        var allowed = Set<Int>()
        for token in 0 ..< backend.vocabSize {
            guard let text = backend.tokenText(token), !text.isEmpty else { continue }
            if text.allSatisfy({ $0.isNumber || $0 == "-" || $0 == "." }),
                text.contains(where: { $0.isNumber })
            {
                allowed.insert(token)
            }
        }
        return allowed
    }

    private mutating func emit(_ text: String) throws -> String {
        for token in try backend.tokenize(text) {
            guard backend.remainingTokens > 0 else {
                throw ConstrainedGenerationError.tokenBudgetExceeded
            }
            try backend.decode(token)
        }
        emittedText += text
        return text
    }

    private func maxFreeStringTokens() -> Int {
        let perStringLimit = max(32, backend.totalTokenBudget / 4)
        return min(backend.remainingTokens, perStringLimit)
    }

    private mutating func generateFreeString(maxTokens: Int) throws -> String {
        var result = ""
        var generated = 0

        while backend.remainingTokens > 0, generated < maxTokens {
            let allowed = result.isEmpty ? stringInitialAllowedTokens : stringContinuationAllowedTokens
            let token = try backend.sample(from: allowed)
            if backend.endTokens.contains(token) {
                throw ConstrainedGenerationError.earlyTermination(emittedText)
            }
            if token == quoteToken { break }

            let text = backend.tokenText(token) ?? ""
            result += text
            emittedText += text
            generated += 1
            try backend.decode(token)
        }

        return result
    }

    private mutating func generateChoice(_ candidates: [String]) throws -> String {
        guard !candidates.isEmpty else {
            throw ConstrainedGenerationError.tokenizationFailed
        }

        let tokenized = try candidates.map { try backend.tokenize($0) }
        for (candidate, tokens) in zip(candidates, tokenized) {
            if candidate.isEmpty { continue }
            if tokens.isEmpty {
                throw ConstrainedGenerationError.tokenizationFailed
            }
        }

        let hasEmptyCandidate = candidates.contains("")
        let hasPrefixCollision = Self.hasPrefixCollision(tokenized: tokenized)
        if hasEmptyCandidate || hasPrefixCollision {
            let chosen = deterministicChoice(from: candidates)
            if !chosen.isEmpty {
                _ = try emit(chosen)
            }
            return chosen
        }

        var prefixes = tokenized
        var emitted = ""
        var position = 0

        while backend.remainingTokens > 0 {
            if prefixes.contains(where: { $0.count == position }) { break }

            let allowed = Set(
                prefixes.compactMap { tokens -> Int? in
                    guard position < tokens.count else { return nil }
                    return tokens[position]
                }
            )

            let token = try backend.sample(from: allowed)
            if backend.endTokens.contains(token) {
                throw ConstrainedGenerationError.earlyTermination(emittedText)
            }
            let text = backend.tokenText(token) ?? ""
            emitted += text
            emittedText += text
            try backend.decode(token)

            prefixes = prefixes.filter { $0.count > position && $0[position] == token }
            position += 1
            if prefixes.isEmpty { break }
        }

        return emitted
    }

    private mutating func generateNumber(_ node: GenerationSchema.NumberNode) throws -> String {
        let allowedTokens = node.integerOnly ? integerTerminators : doubleTerminators
        var result = ""
        let maxTokens = 16
        var generatedTokens = 0

        while backend.remainingTokens > 0, generatedTokens < maxTokens {
            let token = try backend.sample(from: allowedTokens)
            if backend.endTokens.contains(token) {
                throw ConstrainedGenerationError.earlyTermination(emittedText)
            }
            if basicTerminators.contains(token) { break }

            guard let text = backend.tokenText(token) else { break }
            result += text
            emittedText += text
            generatedTokens += 1
            try backend.decode(token)
        }

        guard !result.isEmpty else {
            throw ConstrainedGenerationError.numberOutOfRange("Missing number value")
        }
        return try validateNumberString(result, node: node)
    }

    private func validateNumberString(_ text: String, node: GenerationSchema.NumberNode) throws -> String {
        if node.integerOnly {
            guard let value = Int(text) else {
                throw ConstrainedGenerationError.numberOutOfRange("Invalid integer: \(text)")
            }
            if let minimum = node.minimum, Double(value) < minimum {
                throw ConstrainedGenerationError.numberOutOfRange("Integer \(value) is below minimum \(minimum)")
            }
            if let maximum = node.maximum, Double(value) > maximum {
                throw ConstrainedGenerationError.numberOutOfRange("Integer \(value) exceeds maximum \(maximum)")
            }
            return text
        } else {
            guard let value = Double(text), !value.isNaN, value.isFinite else {
                throw ConstrainedGenerationError.numberOutOfRange("Invalid number: \(text)")
            }
            if let minimum = node.minimum, value < minimum {
                throw ConstrainedGenerationError.numberOutOfRange("Number \(value) is below minimum \(minimum)")
            }
            if let maximum = node.maximum, value > maximum {
                throw ConstrainedGenerationError.numberOutOfRange("Number \(value) exceeds maximum \(maximum)")
            }
            return text
        }
    }

    private mutating func generateNode(_ node: GenerationSchema.Node) throws -> String {
        guard backend.remainingTokens > 0 else {
            throw ConstrainedGenerationError.tokenBudgetExceeded
        }

        switch node {
        case .object(let objectNode):
            return try generateObject(objectNode)
        case .array(let arrayNode):
            return try generateArray(arrayNode)
        case .string(let stringNode):
            return try generateString(stringNode)
        case .number(let numberNode):
            return try generateNumber(numberNode)
        case .boolean:
            return try generateChoice(["true", "false"])
        case .ref(let typeName):
            guard let referenced = schema.defs[typeName] else {
                throw ConstrainedGenerationError.missingReference(typeName)
            }
            return try generateNode(referenced)
        case .anyOf(let variants):
            guard !variants.isEmpty else {
                throw ConstrainedGenerationError.emptyAnyOf
            }
            if variants.count == 1 {
                return try generateNode(variants[0])
            }
            let chosenIndex = backend.remainingTokens % variants.count
            return try generateNode(variants[chosenIndex])
        }
    }

    private mutating func generateObject(_ node: GenerationSchema.ObjectNode) throws -> String {
        let keys = node.properties.keys.sorted()
        let includedKeys = keys.filter { shouldIncludeOptionalProperty($0, required: node.required) }
        var output = try emit("{")

        for (index, key) in includedKeys.enumerated() {
            output += try emit("\"\(key)\":")
            output += try generateNode(node.properties[key] ?? .string(.init()))

            if index < includedKeys.count - 1 {
                output += try emit(",")
            }
        }

        output += try emit("}")
        return output
    }

    private mutating func generateArray(_ node: GenerationSchema.ArrayNode) throws -> String {
        let defaultCount = 4
        let count: Int

        if let minItems = node.minItems, let maxItems = node.maxItems {
            if minItems > maxItems {
                throw ConstrainedGenerationError.invalidArrayBounds(
                    "Minimum items \(minItems) exceeds maximum \(maxItems)"
                )
            }
            let rangeSize = maxItems - minItems + 1
            let offset = rangeSize > 0 ? backend.remainingTokens % rangeSize : 0
            count = minItems + offset
        } else if let minItems = node.minItems {
            count = minItems
        } else if let maxItems = node.maxItems {
            count = maxItems
        } else {
            count = defaultCount
        }
        var output = try emit("[")

        for index in 0 ..< count {
            output += try generateNode(node.items)
            if index < count - 1 {
                output += try emit(",")
            }
        }

        output += try emit("]")
        return output
    }

    private mutating func generateString(_ node: GenerationSchema.StringNode) throws -> String {
        var output = try emit("\"")
        let content: String

        if let choices = node.enumChoices, !choices.isEmpty {
            content = try generateChoice(choices)
        } else {
            content = try generateFreeString(maxTokens: maxFreeStringTokens())
        }

        if let pattern = node.pattern {
            if !matchesPattern(content, pattern: pattern) {
                throw ConstrainedGenerationError.patternMismatch(
                    "Value '\(content)' does not match pattern '\(pattern)'"
                )
            }
        }

        output += content
        output += try emit("\"")
        return output
    }

    private func shouldIncludeOptionalProperty(_ key: String, required: Set<String>) -> Bool {
        if required.contains(key) { return true }
        let minimumBudget = max(8, backend.totalTokenBudget / 10)
        guard backend.remainingTokens > minimumBudget else { return false }
        let hash = key.utf8.reduce(0) { ($0 &* 31) &+ Int($1) }
        return (hash ^ backend.remainingTokens) % 2 == 0
    }

    private func deterministicChoice(from candidates: [String]) -> String {
        guard !candidates.isEmpty else { return "" }
        let index = abs(backend.remainingTokens) % candidates.count
        return candidates[index]
    }

    private static func hasPrefixCollision(tokenized: [[Int]]) -> Bool {
        for (index, candidate) in tokenized.enumerated() {
            for (otherIndex, other) in tokenized.enumerated() where otherIndex != index {
                guard candidate.count < other.count else { continue }
                if Array(other.prefix(candidate.count)) == candidate {
                    return true
                }
            }
        }
        return false
    }

    private func matchesPattern(_ value: String, pattern: String) -> Bool {
        do {
            let regex = try NSRegularExpression(pattern: pattern)
            let range = NSRange(value.startIndex..., in: value)
            return regex.firstMatch(in: value, range: range) != nil
        } catch {
            return false
        }
    }
}

// MARK: - Errors

/// An error that can occur during constrained JSON generation.
package enum ConstrainedGenerationError: LocalizedError {
    /// A required value failed to tokenize.
    case tokenizationFailed

    /// The generation exceeded the available token budget.
    case tokenBudgetExceeded

    /// The tokenizer does not support a required single-token encoding.
    ///
    /// The associated value contains a user-facing description.
    case unsupportedTokenizer(String)

    /// The generated value does not match the required pattern.
    ///
    /// The associated value contains a user-facing description.
    case patternMismatch(String)

    /// The generated number violates numeric bounds or is invalid.
    ///
    /// The associated value contains a user-facing description.
    case numberOutOfRange(String)

    /// The backend emitted an end token before completion.
    ///
    /// The associated value contains the partial output.
    case earlyTermination(String)

    /// The array bounds are invalid.
    ///
    /// The associated value contains a user-facing description.
    case invalidArrayBounds(String)

    /// A referenced schema definition is missing.
    case missingReference(String)

    /// An any-of schema has no choices.
    case emptyAnyOf

    package var errorDescription: String? {
        switch self {
        case .tokenizationFailed:
            return "Failed to tokenize a required value"
        case .tokenBudgetExceeded:
            return "Generation exceeded the available token budget"
        case .unsupportedTokenizer(let details):
            return details
        case .patternMismatch(let details):
            return details
        case .numberOutOfRange(let details):
            return details
        case .earlyTermination:
            return "End token was generated before completion"
        case .invalidArrayBounds(let details):
            return details
        case .missingReference(let name):
            return "Missing referenced schema definition '\(name)'"
        case .emptyAnyOf:
            return "Any-of schema has no choices"
        }
    }
}
