import Foundation

// MARK: - Token Backend

/// Abstracts token-level operations for structured JSON generation.
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

/// Generates JSON conforming to a schema using constrained token sampling.
package struct ConstrainedJSONGenerator<Backend: TokenBackend> {
    private var backend: Backend
    private let schema: GenerationSchema

    private let quoteToken: Int

    private let stringTerminators: Set<Int>
    private let stringInitialAllowedTokens: Set<Int>
    private let stringContinuationAllowedTokens: Set<Int>

    private let basicTerminators: Set<Int>
    private let integerTerminators: Set<Int>
    private let doubleTerminators: Set<Int>

    package init(backend: Backend, schema: GenerationSchema) throws {
        self.backend = backend
        self.schema = schema

        guard let quoteToken = try backend.tokenize("\"").first else {
            throw ConstrainedGenerationError.tokenizationFailed
        }
        self.quoteToken = quoteToken

        self.stringTerminators = backend.endTokens.union([quoteToken])

        var structuralTerminators = backend.endTokens
        for structuralText in [",", "}", "]", ":"] {
            if let token = try backend.tokenize(structuralText).first {
                structuralTerminators.insert(token)
            }
        }
        self.basicTerminators = structuralTerminators
        self.integerTerminators = Self.buildValidIntegerTokens(backend: backend).union(structuralTerminators)
        self.doubleTerminators = Self.buildValidDecimalTokens(backend: backend).union(structuralTerminators)

        let stringContentTokens = Self.buildValidStringTokens(backend: backend)
        self.stringInitialAllowedTokens = stringContentTokens
        self.stringContinuationAllowedTokens = stringContentTokens.union(stringTerminators)
    }

    package mutating func generate() throws -> String {
        try generateNode(schema.root)
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
            if stringTerminators.contains(token) { break }

            var text = backend.tokenText(token) ?? ""
            if result.last?.isWhitespace == true && text.first?.isWhitespace == true {
                text = String(text.drop(while: { $0.isWhitespace }))
            }
            result += text
            generated += 1
            try backend.decode(token)
        }

        return result
    }

    private mutating func generateChoice(_ candidates: [String]) throws -> String {
        let tokenized = try candidates.map { try backend.tokenize($0) }.filter { !$0.isEmpty }
        guard !tokenized.isEmpty else {
            throw ConstrainedGenerationError.tokenizationFailed
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
            emitted += backend.tokenText(token) ?? ""
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
            if basicTerminators.contains(token) { break }

            guard let text = backend.tokenText(token) else { break }
            result += text
            generatedTokens += 1
            try backend.decode(token)
        }

        return clampNumberString(result.isEmpty ? "0" : result, node: node)
    }

    private func clampNumberString(_ text: String, node: GenerationSchema.NumberNode) -> String {
        if node.integerOnly {
            let value = Int(text) ?? 0
            let clamped = clampInt(value, min: node.minimum, max: node.maximum)
            return String(clamped)
        } else {
            let value = Double(text) ?? 0
            let clamped = clampDouble(value, min: node.minimum, max: node.maximum)
            return formatDouble(clamped)
        }
    }

    private func clampInt(_ value: Int, min: Double?, max: Double?) -> Int {
        let lower = min.map { Int(ceil($0)) }
        let upper = max.map { Int(floor($0)) }
        return clamp(value, min: lower, max: upper)
    }

    private func clampDouble(_ value: Double, min: Double?, max: Double?) -> Double {
        clamp(value, min: min, max: max)
    }

    private func clamp<T: Comparable>(_ value: T, min: T?, max: T?) -> T {
        var result = value
        if let min { result = Swift.max(result, min) }
        if let max { result = Swift.min(result, max) }
        return result
    }

    private func formatDouble(_ value: Double) -> String {
        if value.truncatingRemainder(dividingBy: 1) == 0 {
            return String(Int(value))
        }
        let formatted = String(format: "%.6g", value)
        return formatted
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
            if minItems <= maxItems {
                count = Int.random(in: minItems ... maxItems)
            } else {
                count = min(minItems, maxItems)
            }
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

        if let choices = node.enumChoices, !choices.isEmpty {
            output += try generateChoice(choices)
        } else {
            let content = try generateFreeString(maxTokens: maxFreeStringTokens())
            output += content.trimmingCharacters(in: .whitespaces)
        }

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
}

// MARK: - Errors

package enum ConstrainedGenerationError: Error {
    case tokenizationFailed
    case tokenBudgetExceeded
    case missingReference(String)
    case emptyAnyOf
}
