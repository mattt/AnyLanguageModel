import SwiftCompilerPlugin
import SwiftSyntax
import SwiftSyntaxBuilder
import SwiftSyntaxMacros

/// Conforms a type to ``Generable`` protocol.
public struct GenerableMacro: MemberMacro, ExtensionMacro {
    public static func expansion(
        of node: AttributeSyntax,
        providingMembersOf declaration: some DeclGroupSyntax,
        conformingTo protocols: [TypeSyntax],
        in context: some MacroExpansionContext
    ) throws -> [DeclSyntax] {
        if let structDecl = declaration.as(StructDeclSyntax.self) {
            let structName = structDecl.name.text

            let description = extractDescription(from: node)
            let properties = extractGuidedProperties(from: structDecl)

            return [
                generateRawContentProperty(),
                generateMemberwiseInit(properties: properties),
                generateInitFromGeneratedContent(structName: structName, properties: properties),
                generateGeneratedContentProperty(
                    structName: structName,
                    description: description,
                    properties: properties
                ),
                generateGenerationSchemaProperty(
                    structName: structName,
                    description: description,
                    properties: properties
                ),
                generatePartiallyGeneratedStruct(structName: structName, properties: properties),
                generateAsPartiallyGeneratedMethod(structName: structName),
                generateInstructionsRepresentationProperty(),
                generatePromptRepresentationProperty(),
            ]
        } else if let enumDecl = declaration.as(EnumDeclSyntax.self) {
            let enumName = enumDecl.name.text

            let description = extractDescription(from: node)
            let cases = extractEnumCases(from: enumDecl)

            return [
                generateEnumInitFromGeneratedContent(enumName: enumName, cases: cases),
                generateEnumGeneratedContentProperty(
                    enumName: enumName,
                    description: description,
                    cases: cases
                ),
                generateEnumGenerationSchemaProperty(
                    enumName: enumName,
                    description: description,
                    cases: cases
                ),
                generateAsPartiallyGeneratedMethodForEnum(enumName: enumName),
                generateInstructionsRepresentationProperty(),
                generatePromptRepresentationProperty(),
            ]
        } else {
            throw GenerableMacroError.notApplicableToType
        }
    }

    public static func expansion(
        of node: AttributeSyntax,
        attachedTo declaration: some DeclGroupSyntax,
        providingExtensionsOf type: some TypeSyntaxProtocol,
        conformingTo protocols: [TypeSyntax],
        in context: some MacroExpansionContext
    ) throws -> [ExtensionDeclSyntax] {
        let nonisolatedModifier = DeclModifierSyntax(name: .keyword(.nonisolated))

        let extensionDecl = ExtensionDeclSyntax(
            modifiers: DeclModifierListSyntax([nonisolatedModifier]),
            extendedType: type,
            inheritanceClause: InheritanceClauseSyntax(
                inheritedTypes: InheritedTypeListSyntax([
                    InheritedTypeSyntax(type: TypeSyntax(stringLiteral: "Generable"))
                ])
            ),
            memberBlock: MemberBlockSyntax(members: [])
        )
        return [extensionDecl]
    }

    // MARK: - Helpers

    private static func extractDescription(from node: AttributeSyntax) -> String? {
        guard let arguments = node.arguments?.as(LabeledExprListSyntax.self),
            let firstArg = arguments.first,
            firstArg.label?.text == "description",
            let stringLiteral = firstArg.expression.as(StringLiteralExprSyntax.self)
        else {
            return nil
        }
        return stringLiteral.segments.description.trimmingCharacters(in: .init(charactersIn: "\""))
    }

    private static func extractGuidedProperties(from structDecl: StructDeclSyntax) -> [PropertyInfo] {
        var properties: [PropertyInfo] = []

        for member in structDecl.memberBlock.members {
            if let varDecl = member.decl.as(VariableDeclSyntax.self),
                let binding = varDecl.bindings.first,
                let identifier = binding.pattern.as(IdentifierPatternSyntax.self)
            {

                let propertyName = identifier.identifier.text
                let propertyType = binding.typeAnnotation?.type.description ?? "String"

                let guideInfo = extractGuideInfo(from: varDecl.attributes)

                properties.append(
                    PropertyInfo(
                        name: propertyName,
                        type: propertyType,
                        guideDescription: guideInfo.description,
                        guides: guideInfo.guides,
                        pattern: guideInfo.pattern
                    )
                )
            }
        }

        return properties
    }

    private static func extractGuideInfo(from attributes: AttributeListSyntax) -> GuideInfo {
        for attribute in attributes {
            if let attr = attribute.as(AttributeSyntax.self),
                attr.attributeName.description == "Guide"
            {
                if let arguments = attr.arguments?.as(LabeledExprListSyntax.self),
                    let descArg = arguments.first,
                    let stringLiteral = descArg.expression.as(StringLiteralExprSyntax.self)
                {
                    let description = stringLiteral.segments.description.trimmingCharacters(
                        in: .init(charactersIn: "\"")
                    )

                    var guides: [String] = []
                    var pattern: String? = nil

                    for arg in Array(arguments.dropFirst()) {
                        let argText = arg.expression.description

                        if argText.contains(".pattern(") {
                            let patternRegex = #/\.pattern\(\"([^\"]*)\"\)/#
                            if let match = argText.firstMatch(of: patternRegex) {
                                pattern = String(match.1)
                            }
                        } else if argText.contains("pattern(") {
                            let patternRegex = #/pattern\(\"([^\"]*)\"\)/#
                            if let match = argText.firstMatch(of: patternRegex) {
                                pattern = String(match.1)
                            }
                        } else {
                            guides.append(argText)
                        }
                    }

                    return GuideInfo(description: description, guides: guides, pattern: pattern)
                }
            }
        }
        return GuideInfo(description: nil, guides: [], pattern: nil)
    }

    private static func topLevelColonIndex(in text: String) -> String.Index? {
        var totalDepth = 0

        for index in text.indices {
            switch text[index] {
            case "[":
                totalDepth += 1
            case "]":
                totalDepth -= 1
            case "<":
                totalDepth += 1
            case ">":
                totalDepth -= 1
            case "(":
                totalDepth += 1
            case ")":
                totalDepth -= 1
            case ":" where totalDepth == 0:
                return index
            default:
                break
            }

            if totalDepth < 0 {
                return nil
            }
        }

        return nil
    }

    private static func extractDictionaryTypes(_ type: String) -> (key: String, value: String)? {
        let trimmed = type.trimmingCharacters(in: .whitespacesAndNewlines)

        guard trimmed.hasPrefix("[") && trimmed.hasSuffix("]") else {
            return nil
        }

        let inner = String(trimmed.dropFirst().dropLast())
        guard let colonIndex = topLevelColonIndex(in: inner) else {
            return nil
        }

        let key = inner[..<colonIndex].trimmingCharacters(in: .whitespacesAndNewlines)
        let value = inner[inner.index(after: colonIndex)...]
            .trimmingCharacters(in: .whitespacesAndNewlines)

        guard !key.isEmpty && !value.isEmpty else {
            return nil
        }

        return (key: key, value: value)
    }

    private static func isDictionaryType(_ type: String) -> Bool {
        extractDictionaryTypes(type) != nil
    }

    private static func baseTypeName(_ type: String) -> String {
        let trimmed = type.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.hasSuffix("?") {
            return String(trimmed.dropLast()).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return trimmed
    }

    private static func arrayElementType(from type: String) -> String? {
        let trimmed = type.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.hasPrefix("[") && trimmed.hasSuffix("]") {
            let inner = String(trimmed.dropFirst().dropLast())
            guard topLevelColonIndex(in: inner) == nil else {
                return nil
            }
            return inner.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        if trimmed.hasPrefix("Array<") && trimmed.hasSuffix(">") {
            return String(trimmed.dropFirst("Array<".count).dropLast())
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return nil
    }

    private static let primitiveTypes: Set<String> = [
        "String",
        "Int",
        "Double",
        "Float",
        "Bool",
        "Decimal",
    ]

    private static func partiallyGeneratedTypeName(for type: String) -> String {
        partiallyGeneratedTypeName(for: type, preserveOptional: false)
    }

    private static func partiallyGeneratedTypeName(for type: String, preserveOptional: Bool) -> String {
        let trimmed = type.trimmingCharacters(in: .whitespacesAndNewlines)
        if preserveOptional {
            var normalized = trimmed
            var optionalCount = 0
            while normalized.hasSuffix("?") {
                normalized = String(normalized.dropLast())
                optionalCount += 1
            }
            if optionalCount > 1 {
                return "\(partiallyGeneratedTypeName(for: normalized, preserveOptional: false))?"
            }
            if optionalCount == 1 {
                return "\(partiallyGeneratedTypeName(for: normalized, preserveOptional: true))?"
            }
        }

        let baseType = baseTypeName(trimmed)
        if primitiveTypes.contains(baseType) || isDictionaryType(baseType) {
            return baseType
        }
        if let elementType = arrayElementType(from: baseType) {
            let elementPartial = partiallyGeneratedTypeName(for: elementType, preserveOptional: true)
            return "[\(elementPartial)]"
        }
        return "\(baseType).PartiallyGenerated"
    }

    private static func getDefaultValue(for type: String) -> String {
        let trimmedType = type.trimmingCharacters(in: .whitespacesAndNewlines)

        if trimmedType.hasSuffix("?") {
            return "nil"
        }

        if isDictionaryType(trimmedType) {
            return "[:]"
        }

        if trimmedType.hasPrefix("[") && trimmedType.hasSuffix("]") {
            return "[]"
        }

        switch trimmedType {
        case "String":
            return "\"\""
        case "Int":
            return "0"
        case "Double", "Float":
            return "0.0"
        case "Bool":
            return "false"
        default:
            return "nil"
        }
    }

    private static func generatePropertyAssignment(for property: PropertyInfo) -> String {
        let propertyName = property.name
        let propertyType = property.type.trimmingCharacters(in: .whitespacesAndNewlines)
        let defaultValue = getDefaultValue(for: propertyType)

        switch propertyType {
        case "String":
            return
                "self.\(propertyName) = (json[\"\(propertyName)\"] as? String) ?? \(defaultValue)"
        case "Int":
            return "self.\(propertyName) = (json[\"\(propertyName)\"] as? Int) ?? \(defaultValue)"
        case "Double":
            return
                "self.\(propertyName) = (json[\"\(propertyName)\"] as? Double) ?? \(defaultValue)"
        case "Float":
            return
                "self.\(propertyName) = Float((json[\"\(propertyName)\"] as? Double) ?? Double(\(defaultValue)))"
        case "Bool":
            return "self.\(propertyName) = (json[\"\(propertyName)\"] as? Bool) ?? \(defaultValue)"
        default:
            return "self.\(propertyName) = \(defaultValue)"
        }
    }

    private static func generateRawContentProperty() -> DeclSyntax {
        return DeclSyntax(
            stringLiteral: """
                private let _rawGeneratedContent: GeneratedContent
                """
        )
    }

    private static func generateMemberwiseInit(properties: [PropertyInfo]) -> DeclSyntax {
        if properties.isEmpty {
            return DeclSyntax(
                stringLiteral: """
                    nonisolated public init() {
                        self._rawGeneratedContent = GeneratedContent(kind: .structure(properties: [:], orderedKeys: []))
                    }
                    """
            )
        }

        let parameters = properties.map { prop in
            "\(prop.name): \(prop.type)"
        }.joined(separator: ", ")

        let assignments = properties.map { prop in
            "self.\(prop.name) = \(prop.name)"
        }.joined(separator: "\n        ")

        let propertyConversions = properties.map { prop in
            let propName = prop.name
            let propType = prop.type

            if propType.hasSuffix("?") {
                let baseType = String(propType.dropLast())
                if baseType == "String" {
                    return
                        "properties[\"\(propName)\"] = \(propName).map { GeneratedContent($0) } ?? GeneratedContent(kind: .null)"
                } else if baseType == "Int" || baseType == "Double" || baseType == "Float"
                    || baseType == "Bool" || baseType == "Decimal"
                {
                    return
                        "properties[\"\(propName)\"] = \(propName).map { $0.generatedContent } ?? GeneratedContent(kind: .null)"
                } else if isDictionaryType(baseType) {
                    return
                        "properties[\"\(propName)\"] = \(propName).map { $0.generatedContent } ?? GeneratedContent(kind: .null)"
                } else if baseType.hasPrefix("[") && baseType.hasSuffix("]") {
                    return
                        "properties[\"\(propName)\"] = \(propName).map { GeneratedContent(elements: $0) } ?? GeneratedContent(kind: .null)"
                } else {
                    return """
                        if let value = \(propName) {
                                    properties["\(propName)"] = value.generatedContent
                                } else {
                                    properties["\(propName)"] = GeneratedContent(kind: .null)
                                }
                        """
                }
            } else if isDictionaryType(propType) {
                return "properties[\"\(propName)\"] = \(propName).generatedContent"
            } else if propType.hasPrefix("[") && propType.hasSuffix("]") {
                return "properties[\"\(propName)\"] = GeneratedContent(elements: \(propName))"
            } else {
                switch propType {
                case "String":
                    return "properties[\"\(propName)\"] = GeneratedContent(\(propName))"
                case "Int", "Double", "Float", "Bool", "Decimal":
                    return "properties[\"\(propName)\"] = \(propName).generatedContent"
                default:
                    return "properties[\"\(propName)\"] = \(propName).generatedContent"
                }
            }
        }.joined(separator: "\n        ")

        let orderedKeys = properties.map { "\"\($0.name)\"" }.joined(separator: ", ")

        return DeclSyntax(
            stringLiteral: """
                nonisolated public init(\(parameters)) {
                    \(assignments)
                    
                    var properties: [String: GeneratedContent] = [:]
                    \(propertyConversions)
                    
                    self._rawGeneratedContent = GeneratedContent(
                        kind: .structure(
                            properties: properties,
                            orderedKeys: [\(orderedKeys)]
                        )
                    )
                }
                """
        )
    }

    private static func generateInitFromGeneratedContent(
        structName: String,
        properties: [PropertyInfo]
    ) -> DeclSyntax {
        let propertyExtractions = properties.map { prop in
            generatePropertyExtraction(propertyName: prop.name, propertyType: prop.type)
        }.joined(separator: "\n            ")

        if properties.isEmpty {
            return DeclSyntax(
                stringLiteral: """
                    nonisolated public init(_ generatedContent: GeneratedContent) throws {
                        self._rawGeneratedContent = generatedContent

                        guard case .structure = generatedContent.kind else {
                            throw DecodingError.typeMismatch(
                                \(structName).self,
                                DecodingError.Context(codingPath: [], debugDescription: "Expected structure for \(structName)")
                            )
                        }
                    }
                    """
            )
        } else {
            return DeclSyntax(
                stringLiteral: """
                    nonisolated public init(_ generatedContent: GeneratedContent) throws {
                        self._rawGeneratedContent = generatedContent

                        guard case .structure(let properties, _) = generatedContent.kind else {
                            throw DecodingError.typeMismatch(
                                \(structName).self,
                                DecodingError.Context(codingPath: [], debugDescription: "Expected structure for \(structName)")
                            )
                        }

                        \(propertyExtractions)
                    }
                    """
            )
        }
    }

    private static func generatePartialPropertyExtraction(
        propertyName: String,
        propertyType: String
    ) -> String {
        let baseType = baseTypeName(propertyType)

        switch baseType {
        case "String":
            return "self.\(propertyName) = try? properties[\"\(propertyName)\"]?.value(String.self)"
        case "Int":
            return "self.\(propertyName) = try? properties[\"\(propertyName)\"]?.value(Int.self)"
        case "Double":
            return "self.\(propertyName) = try? properties[\"\(propertyName)\"]?.value(Double.self)"
        case "Float":
            return "self.\(propertyName) = try? properties[\"\(propertyName)\"]?.value(Float.self)"
        case "Bool":
            return "self.\(propertyName) = try? properties[\"\(propertyName)\"]?.value(Bool.self)"
        default:
            if isDictionaryType(baseType) {
                return """
                    if let value = properties[\"\(propertyName)\"] {
                        self.\(propertyName) = try? \(baseType)(value)
                    } else {
                        self.\(propertyName) = nil
                    }
                    """
            } else if let elementType = arrayElementType(from: baseType) {
                let elementPartial = partiallyGeneratedTypeName(for: elementType, preserveOptional: true)
                let arrayPartial = "[\(elementPartial)]"
                return """
                    if let value = properties[\"\(propertyName)\"] {
                        self.\(propertyName) = try? \(arrayPartial)(value)
                    } else {
                        self.\(propertyName) = nil
                    }
                    """
            } else {
                let partialType = partiallyGeneratedTypeName(for: baseType)
                return """
                    if let value = properties[\"\(propertyName)\"] {
                        self.\(propertyName) = try? \(partialType)(value)
                    } else {
                        self.\(propertyName) = nil
                    }
                    """
            }
        }
    }

    private static func generatePropertyExtraction(propertyName: String, propertyType: String)
        -> String
    {
        switch propertyType {
        case "String":
            return """
                self.\(propertyName) = try properties["\(propertyName)"]?.value(String.self) ?? ""
                """
        case "Int":
            return """
                self.\(propertyName) = try properties["\(propertyName)"]?.value(Int.self) ?? 0
                """
        case "Double":
            return """
                self.\(propertyName) = try properties["\(propertyName)"]?.value(Double.self) ?? 0.0
                """
        case "Float":
            return """
                self.\(propertyName) = try properties["\(propertyName)"]?.value(Float.self) ?? 0.0
                """
        case "Bool":
            return """
                self.\(propertyName) = try properties["\(propertyName)"]?.value(Bool.self) ?? false
                """
        default:
            let isOptional = propertyType.hasSuffix("?")
            let isDictionary = isDictionaryType(
                propertyType.replacingOccurrences(of: "?", with: "")
            )
            let isArray =
                !isDictionary && propertyType.hasPrefix("[") && propertyType.hasSuffix("]")

            if isOptional {
                let baseType = propertyType.replacingOccurrences(of: "?", with: "")

                if baseType == "Int" || baseType == "String" || baseType == "Double"
                    || baseType == "Float" || baseType == "Bool"
                {
                    return """
                        if let value = properties["\(propertyName)"] {
                            switch value.kind {
                            case .null:
                                self.\(propertyName) = nil
                            default:
                                self.\(propertyName) = try value.value(\(baseType).self)
                            }
                        } else {
                            self.\(propertyName) = nil
                        }
                        """
                } else {
                    return """
                        if let value = properties["\(propertyName)"] {
                            switch value.kind {
                            case .null:
                                self.\(propertyName) = nil
                            default:
                                self.\(propertyName) = try \(baseType)(value)
                            }
                        } else {
                            self.\(propertyName) = nil
                        }
                        """
                }

            } else if isDictionary {
                return """
                    if let value = properties["\(propertyName)"] {
                        self.\(propertyName) = try \(propertyType)(value)
                    } else {
                        self.\(propertyName) = [:]
                    }
                    """
            } else if isArray {
                return """
                    if let value = properties["\(propertyName)"] {
                        self.\(propertyName) = try \(propertyType)(value)
                    } else {
                        self.\(propertyName) = []
                    }
                    """
            } else {
                return """
                    if let value = properties["\(propertyName)"] {
                        self.\(propertyName) = try \(propertyType)(value)
                    } else {
                        self.\(propertyName) = try \(propertyType)(GeneratedContent("{}"))
                    }
                    """
            }
        }
    }

    private static func generateGeneratedContentProperty(
        structName: String,
        description: String?,
        properties: [PropertyInfo]
    ) -> DeclSyntax {
        let propertyConversions = properties.map { prop in
            let propName = prop.name
            let propType = prop.type

            if propType.hasSuffix("?") {
                let baseType = String(propType.dropLast())
                if baseType == "String" {
                    return
                        "properties[\"\(propName)\"] = \(propName).map { GeneratedContent($0) } ?? GeneratedContent(kind: .null)"
                } else if baseType == "Int" || baseType == "Double" || baseType == "Float"
                    || baseType == "Bool" || baseType == "Decimal"
                {
                    return
                        "properties[\"\(propName)\"] = \(propName).map { $0.generatedContent } ?? GeneratedContent(kind: .null)"
                } else if isDictionaryType(baseType) {
                    return
                        "properties[\"\(propName)\"] = \(propName).map { $0.generatedContent } ?? GeneratedContent(kind: .null)"
                } else if baseType.hasPrefix("[") && baseType.hasSuffix("]") {
                    return
                        "properties[\"\(propName)\"] = \(propName).map { GeneratedContent(elements: $0) } ?? GeneratedContent(kind: .null)"
                } else {
                    return """
                        if let value = \(propName) {
                                    properties["\(propName)"] = value.generatedContent
                                } else {
                                    properties["\(propName)"] = GeneratedContent(kind: .null)
                                }
                        """
                }
            } else if isDictionaryType(propType) {
                return "properties[\"\(propName)\"] = \(propName).generatedContent"
            } else if propType.hasPrefix("[") && propType.hasSuffix("]") {
                return "properties[\"\(propName)\"] = GeneratedContent(elements: \(propName))"
            } else {
                switch propType {
                case "String":
                    return "properties[\"\(propName)\"] = GeneratedContent(\(propName))"
                case "Int", "Double", "Float", "Bool", "Decimal":
                    return "properties[\"\(propName)\"] = \(propName).generatedContent"
                default:
                    return "properties[\"\(propName)\"] = \(propName).generatedContent"
                }
            }
        }.joined(separator: "\n            ")

        let orderedKeys = properties.map { "\"\($0.name)\"" }.joined(separator: ", ")

        if properties.isEmpty {
            return DeclSyntax(
                stringLiteral: """
                    nonisolated public var generatedContent: GeneratedContent {
                        let properties: [String: GeneratedContent] = [:]

                        return GeneratedContent(
                            kind: .structure(
                                properties: properties,
                                orderedKeys: []
                            )
                        )
                    }
                    """
            )
        } else {
            return DeclSyntax(
                stringLiteral: """
                    nonisolated public var generatedContent: GeneratedContent {
                        var properties: [String: GeneratedContent] = [:]
                        \(propertyConversions)

                        return GeneratedContent(
                            kind: .structure(
                                properties: properties,
                                orderedKeys: [\(orderedKeys)]
                            )
                        )
                    }
                    """
            )
        }
    }

    private static func generateGenerationSchemaProperty(
        structName: String,
        description: String?,
        properties: [PropertyInfo]
    ) -> DeclSyntax {
        let propertySchemas = properties.map { prop in
            var guidesArray = "[]"
            if !prop.guides.isEmpty || prop.pattern != nil {
                var guides: [String] = []

                if let pattern = prop.pattern {
                    guides.append(".pattern(\"\(pattern)\")")
                }

                guides.append(contentsOf: prop.guides)
                guidesArray = "[\(guides.joined(separator: ", "))]"
            }

            // Escape the description string so it can be safely embedded in generated code.
            // Multi-line strings need newlines converted to \n escape sequences,
            // and special characters (backslashes, quotes) must be escaped.
            let escapedDescription: String
            if let desc = prop.guideDescription {
                let escaped =
                    desc
                    .replacingOccurrences(of: "\\", with: "\\\\")  // Escape backslashes first
                    .replacingOccurrences(of: "\"", with: "\\\"")  // Escape quotes
                    .replacingOccurrences(of: "\n", with: "\\n")  // Convert newlines to escape sequences
                escapedDescription = "\"\(escaped)\""
            } else {
                escapedDescription = "nil"
            }

            return """
                GenerationSchema.Property(
                                name: "\(prop.name)",
                                description: \(escapedDescription),
                                type: \(prop.type).self,
                                guides: \(guidesArray)
                            )
                """
        }.joined(separator: ",\n            ")

        return DeclSyntax(
            stringLiteral: """
                nonisolated public static var generationSchema: GenerationSchema {
                    return GenerationSchema(
                        type: Self.self,
                        description: \(description.map { "\"\($0)\"" } ?? "\"Generated \(structName)\""),
                        properties: [\(properties.isEmpty ? "" : "\n            \(propertySchemas)\n        ")]
                    )
                }
                """
        )
    }

    private static func generateAsPartiallyGeneratedMethod(structName: String) -> DeclSyntax {
        return DeclSyntax(
            stringLiteral: """
                nonisolated public func asPartiallyGenerated() -> PartiallyGenerated {
                    return try! PartiallyGenerated(_rawGeneratedContent)
                }
                """
        )
    }

    private static func generateAsPartiallyGeneratedMethodForEnum(enumName: String) -> DeclSyntax {
        return DeclSyntax(
            stringLiteral: """
                nonisolated public func asPartiallyGenerated() -> \(enumName) {
                    return self
                }
                """
        )
    }

    private static func generatePartiallyGeneratedStruct(
        structName: String,
        properties: [PropertyInfo]
    ) -> DeclSyntax {
        let optionalProperties = properties.map { prop in
            let partialType = partiallyGeneratedTypeName(for: prop.type)
            return "public let \(prop.name): \(partialType)?"
        }.joined(separator: "\n        ")

        let propertyExtractions = properties.map { prop in
            generatePartialPropertyExtraction(propertyName: prop.name, propertyType: prop.type)
        }.joined(separator: "\n            ")

        return DeclSyntax(
            stringLiteral: """
                public struct PartiallyGenerated: Identifiable, Sendable, ConvertibleFromGeneratedContent {
                    public var id: GenerationID

                    \(optionalProperties)

                    private let rawContent: GeneratedContent

                    public init(_ generatedContent: GeneratedContent) throws {
                        self.id = generatedContent.id ?? GenerationID()
                        self.rawContent = generatedContent

                        if \(properties.isEmpty ? "case .structure = generatedContent.kind" : "case .structure(let properties, _) = generatedContent.kind") {
                            \(propertyExtractions)
                        } else {
                            \(properties.map { "self.\($0.name) = nil" }.joined(separator: "\n                    "))
                        }
                    }

                    public var generatedContent: GeneratedContent {
                        return rawContent
                    }
                }
                """
        )
    }

    private static func generateInstructionsRepresentationProperty() -> DeclSyntax {
        return DeclSyntax(
            stringLiteral: """
                nonisolated public var instructionsRepresentation: Instructions {
                    return Instructions(self.generatedContent.jsonString)
                }
                """
        )
    }

    private static func generatePromptRepresentationProperty() -> DeclSyntax {
        return DeclSyntax(
            stringLiteral: """
                nonisolated public var promptRepresentation: Prompt {
                    return Prompt(self.generatedContent.jsonString)
                }
                """
        )
    }

    private static func extractEnumCases(from enumDecl: EnumDeclSyntax) -> [EnumCaseInfo] {
        var cases: [EnumCaseInfo] = []

        for member in enumDecl.memberBlock.members {
            if let caseDecl = member.decl.as(EnumCaseDeclSyntax.self) {
                for element in caseDecl.elements {
                    let caseName = element.name.text
                    var associatedValues: [(label: String?, type: String)] = []

                    if let parameterClause = element.parameterClause {
                        for parameter in parameterClause.parameters {
                            let label = parameter.firstName?.text
                            let type = parameter.type.description.trimmingCharacters(
                                in: .whitespacesAndNewlines
                            )
                            associatedValues.append((label: label, type: type))
                        }
                    }

                    let guideDescription: String? = nil

                    cases.append(
                        EnumCaseInfo(
                            name: caseName,
                            associatedValues: associatedValues,
                            guideDescription: guideDescription
                        )
                    )
                }
            }
        }

        return cases
    }

    private static func generateEnumInitFromGeneratedContent(
        enumName: String,
        cases: [EnumCaseInfo]
    ) -> DeclSyntax {
        let hasAnyAssociatedValues = cases.contains { $0.hasAssociatedValues }

        if hasAnyAssociatedValues {
            let switchCases = cases.map { enumCase in
                if enumCase.associatedValues.isEmpty {
                    return """
                        case "\(enumCase.name)":
                            self = .\(enumCase.name)
                        """
                } else if enumCase.isSingleUnlabeledValue {
                    let valueType = enumCase.associatedValues[0].type
                    return generateSingleValueCase(caseName: enumCase.name, valueType: valueType)
                } else {
                    return generateMultipleValueCase(
                        caseName: enumCase.name,
                        associatedValues: enumCase.associatedValues
                    )
                }
            }.joined(separator: "\n                ")

            return DeclSyntax(
                stringLiteral: """
                    nonisolated public init(_ generatedContent: GeneratedContent) throws {
                        do {
                            guard case .structure(let properties, _) = generatedContent.kind else {
                                throw DecodingError.typeMismatch(
                                    \(enumName).self,
                                    DecodingError.Context(codingPath: [], debugDescription: "Expected structure for enum \(enumName)")
                                )
                            }

                            guard case .string(let caseValue) = properties["case"]?.kind else {
                                struct Key: CodingKey {
                                    var stringValue: String
                                    var intValue: Int? { nil }
                                    init(stringValue: String) { self.stringValue = stringValue }
                                    init?(intValue: Int) { nil }
                                }
                                throw DecodingError.keyNotFound(
                                    Key(stringValue: "case"),
                                    DecodingError.Context(codingPath: [], debugDescription: "Missing 'case' property in enum data for \(enumName)")
                                )
                            }

                            let valueContent = properties["value"]

                            switch caseValue {
                            \(switchCases)
                            default:
                                throw DecodingError.dataCorrupted(
                                    DecodingError.Context(codingPath: [], debugDescription: "Invalid enum case '\\(caseValue)' for \(enumName). Valid cases: [\(cases.map { $0.name }.joined(separator: ", "))]")
                                )
                            }
                        } catch {
                            guard case .string(let value) = generatedContent.kind else {
                                throw error
                            }
                            let trimmedValue = value.trimmingCharacters(in: .whitespacesAndNewlines)
                            switch trimmedValue {
                            \(cases.filter { !$0.hasAssociatedValues }.map { "case \"\($0.name)\": self = .\($0.name)" }.joined(separator: "\n                    "))
                            default:
                                throw DecodingError.dataCorrupted(
                                    DecodingError.Context(codingPath: [], debugDescription: "Invalid enum case '\\(trimmedValue)' for \(enumName). Valid cases: [\(cases.map { $0.name }.joined(separator: ", "))]")
                                )
                            }
                        }
                    }
                    """
            )
        } else {
            let switchCases = cases.map { enumCase in
                "case \"\(enumCase.name)\": self = .\(enumCase.name)"
            }.joined(separator: "\n            ")

            return DeclSyntax(
                stringLiteral: """
                    nonisolated public init(_ generatedContent: GeneratedContent) throws {
                        guard case .string(let value) = generatedContent.kind else {
                            throw DecodingError.typeMismatch(
                                \(enumName).self,
                                DecodingError.Context(codingPath: [], debugDescription: "Expected string for enum \(enumName)")
                            )
                        }
                        let trimmedValue = value.trimmingCharacters(in: .whitespacesAndNewlines)

                        switch trimmedValue {
                        \(switchCases)
                        default:
                            throw DecodingError.dataCorrupted(
                                DecodingError.Context(codingPath: [], debugDescription: "Invalid enum case '\\(trimmedValue)' for \(enumName). Valid cases: [\(cases.map { $0.name }.joined(separator: ", "))]")
                            )
                        }
                    }
                    """
            )
        }
    }

    private static func generateSingleValueCase(caseName: String, valueType: String) -> String {
        switch valueType {
        case "String":
            return """
                case "\(caseName)":
                    if let valueContent = valueContent,
                       case .string(let stringValue) = valueContent.kind {
                        self = .\(caseName)(stringValue)
                    } else {
                        self = .\(caseName)("")
                    }
                """
        case "Int":
            return """
                case "\(caseName)":
                    if let valueContent = valueContent {
                        let intValue = try valueContent.value(Int.self)
                        self = .\(caseName)(intValue)
                    } else {
                        self = .\(caseName)(0)
                    }
                """
        case "Double":
            return """
                case "\(caseName)":
                    if let valueContent = valueContent {
                        let doubleValue = try valueContent.value(Double.self)
                        self = .\(caseName)(doubleValue)
                    } else {
                        self = .\(caseName)(0.0)
                    }
                """
        case "Bool":
            return """
                case "\(caseName)":
                    if let valueContent = valueContent {
                        let boolValue = try valueContent.value(Bool.self)
                        self = .\(caseName)(boolValue)
                    } else {
                        self = .\(caseName)(false)
                    }
                """
        default:
            return """
                case "\(caseName)":
                    if let valueContent = valueContent {
                        let associatedValue = try \(valueType)(valueContent)
                        self = .\(caseName)(associatedValue)
                    } else {
                        throw DecodingError.valueNotFound(
                            \(valueType).self,
                            DecodingError.Context(codingPath: [], debugDescription: "Missing value for enum case '\(caseName)' with associated type \(valueType)")
                        )
                    }
                """
        }
    }

    private static func generateMultipleValueCase(
        caseName: String,
        associatedValues: [(label: String?, type: String)]
    ) -> String {
        let valueExtractions = associatedValues.enumerated().map { index, assocValue in
            let label = assocValue.label ?? "param\(index)"
            let type = assocValue.type

            switch type {
            case "String":
                return "let \(label) = try valueProperties[\"\(label)\"]?.value(String.self) ?? \"\""
            case "Int":
                return "let \(label) = try valueProperties[\"\(label)\"]?.value(Int.self) ?? 0"
            case "Double":
                return "let \(label) = try valueProperties[\"\(label)\"]?.value(Double.self) ?? 0.0"
            case "Bool":
                return "let \(label) = try valueProperties[\"\(label)\"]?.value(Bool.self) ?? false"
            default:
                return
                    "let \(label) = try \(type)(valueProperties[\"\(label)\"] ?? GeneratedContent(\"{}\"))"
            }
        }.joined(separator: "\n                    ")

        let parameterList = associatedValues.enumerated().map { index, assocValue in
            let label = assocValue.label ?? "param\(index)"
            if assocValue.label != nil {
                return "\(label): \(label)"
            } else {
                return label
            }
        }.joined(separator: ", ")

        return """
            case "\(caseName)":
                if let valueContent = valueContent {
                    guard case .structure(let valueProperties, _) = valueContent.kind else {
                        throw DecodingError.typeMismatch(
                            [String: Any].self,
                            DecodingError.Context(codingPath: [], debugDescription: "Expected structure for enum case '\(caseName)' associated values")
                        )
                    }
                    \(valueExtractions)
                    self = .\(caseName)(\(parameterList))
                } else {
                    throw DecodingError.valueNotFound(
                        [String: Any].self,
                        DecodingError.Context(codingPath: [], debugDescription: "Missing value data for enum case '\(caseName)' with associated values")
                    )
                }
            """
    }

    private static func generateEnumGeneratedContentProperty(
        enumName: String,
        description: String?,
        cases: [EnumCaseInfo]
    ) -> DeclSyntax {
        let hasAnyAssociatedValues = cases.contains { $0.hasAssociatedValues }

        if hasAnyAssociatedValues {
            let switchCases = cases.map { enumCase in
                if enumCase.associatedValues.isEmpty {
                    return """
                        case .\(enumCase.name):
                            return GeneratedContent(properties: [
                                "case": GeneratedContent("\(enumCase.name)"),
                                "value": GeneratedContent("")
                            ])
                        """
                } else if enumCase.isSingleUnlabeledValue {
                    return """
                        case .\\(enumCase.name)(let value):
                            return GeneratedContent(properties: [
                                "case": GeneratedContent("\\(enumCase.name)"),
                                "value": GeneratedContent("\\\\(value)")
                            ])
                        """
                } else {
                    return generateMultipleValueSerialization(
                        caseName: enumCase.name,
                        associatedValues: enumCase.associatedValues
                    )
                }
            }.joined(separator: "\n            ")

            return DeclSyntax(
                stringLiteral: """
                    nonisolated public var generatedContent: GeneratedContent {
                        switch self {
                        \(switchCases)
                        }
                    }
                    """
            )
        } else {
            let switchCases = cases.map { enumCase in
                "case .\(enumCase.name): return GeneratedContent(\"\(enumCase.name)\")"
            }.joined(separator: "\n            ")

            return DeclSyntax(
                stringLiteral: """
                    nonisolated public var generatedContent: GeneratedContent {
                        switch self {
                        \(switchCases)
                        }
                    }
                    """
            )
        }
    }

    private static func generateSingleValueSerialization(caseName: String, valueType: String)
        -> String
    {
        switch valueType {
        case "String", "Int", "Double", "Bool":
            return """
                case .\(caseName)(let value):
                    return GeneratedContent(properties: [
                        "case": GeneratedContent("\(caseName)"),
                        "value": GeneratedContent("\\(value)")
                    ])
                """
        default:
            return """
                case .\(caseName)(let value):
                    return GeneratedContent(properties: [
                        "case": GeneratedContent("\(caseName)"),
                        "value": value.generatedContent
                    ])
                """
        }
    }

    private static func generateMultipleValueSerialization(
        caseName: String,
        associatedValues: [(label: String?, type: String)]
    ) -> String {
        let parameterList = associatedValues.enumerated().map { index, assocValue in
            let label = assocValue.label ?? "param\(index)"
            return "let \(label)"
        }.joined(separator: ", ")

        let propertyMappings = associatedValues.enumerated().map { index, assocValue in
            let label = assocValue.label ?? "param\(index)"
            let type = assocValue.type

            switch type {
            case "String", "Int", "Double", "Bool":
                return "\"\(label)\": GeneratedContent(\"\\(\(label))\")"
            default:
                return "\"\(label)\": \(label).generatedContent"
            }
        }.joined(separator: ",\n                        ")

        return """
            case .\(caseName)(\(parameterList)):
                return GeneratedContent(properties: [
                    "case": GeneratedContent("\(caseName)"),
                    "value": GeneratedContent(properties: [
                        \(propertyMappings)
                    ])
                ])
            """
    }

    private static func generateEnumFromGeneratedContentMethod(enumName: String) -> DeclSyntax {
        return DeclSyntax(
            stringLiteral: """
                public static func from(generatedContent: GeneratedContent) throws -> \(enumName) {
                    return try \(enumName)(generatedContent)
                }
                """
        )
    }

    private static func generateEnumGenerationSchemaProperty(
        enumName: String,
        description: String?,
        cases: [EnumCaseInfo]
    ) -> DeclSyntax {
        let hasAnyAssociatedValues = cases.contains { $0.hasAssociatedValues }

        if hasAnyAssociatedValues {

            let caseProperty = """
                GenerationSchema.Property(
                                        name: "case",
                                        description: "Enum case identifier",
                                        type: String.self,
                                        guides: []
                                    )
                """
            let valueProperty = """
                GenerationSchema.Property(
                                        name: "value",
                                        description: "Associated value data",
                                        type: String.self,
                                        guides: []
                                    )
                """

            return DeclSyntax(
                stringLiteral: """
                    nonisolated public static var generationSchema: GenerationSchema {
                        return GenerationSchema(
                            type: Self.self,
                            description: \(description.map { "\"\($0)\"" } ?? "\"Generated \(enumName)\""),
                            properties: [
                                \(caseProperty),
                                \(valueProperty)
                            ]
                        )
                    }
                    """
            )
        } else {
            let caseNames = cases.map { "\"\($0.name)\"" }.joined(separator: ", ")

            return DeclSyntax(
                stringLiteral: """
                    nonisolated public static var generationSchema: GenerationSchema {
                        return GenerationSchema(
                            type: Self.self,
                            description: \(description.map { "\"\($0)\"" } ?? "\"Generated \(enumName)\""),
                            anyOf: [\(caseNames)]
                        )
                    }
                    """
            )
        }
    }
}

// MARK: - Error

public enum GenerableMacroError: Error, CustomStringConvertible {
    case notApplicableToType
    case invalidSyntax
    case missingRequiredParameter

    public var description: String {
        switch self {
        case .notApplicableToType:
            return "@Generable can only be applied to structs, actors, or enumerations"
        case .invalidSyntax:
            return "Invalid macro syntax"
        case .missingRequiredParameter:
            return "Missing required parameter"
        }
    }
}

// MARK: -

private struct EnumCaseInfo {
    let name: String
    let associatedValues: [(label: String?, type: String)]
    let guideDescription: String?

    var hasAssociatedValues: Bool {
        !associatedValues.isEmpty
    }

    var isSingleUnlabeledValue: Bool {
        associatedValues.count == 1 && associatedValues[0].label == nil
    }

    var isMultipleLabeledValues: Bool {
        associatedValues.count > 1
            || (associatedValues.count == 1 && associatedValues[0].label != nil)
    }
}

private struct GuideInfo {
    let description: String?
    let guides: [String]
    let pattern: String?
}

private struct PropertyInfo {
    let name: String
    let type: String
    let guideDescription: String?
    let guides: [String]
    let pattern: String?
}
