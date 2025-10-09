import protocol Foundation.LocalizedError
import class Foundation.JSONEncoder
import class Foundation.JSONDecoder
import struct Foundation.Decimal

/// A type that describes the properties of an object and any guides
/// on their values.
///
/// Generation  schemas guide the output of a ``SystemLanguageModel`` to deterministically
/// ensure the output is in the desired format.
public struct GenerationSchema: Sendable, Codable, CustomDebugStringConvertible {

    // MARK: - Structure

    indirect enum Node: Sendable, Codable {
        case object(ObjectNode)
        case array(ArrayNode)
        case string(StringNode)
        case number(NumberNode)
        case boolean
        case anyOf([Node])
        case ref(String)

        private enum CodingKeys: String, CodingKey {
            case type, properties, required, additionalProperties
            case items, minItems, maxItems
            case pattern, `enum`, anyOf
            case ref = "$ref"
            case description
            case minimum, maximum
        }

        func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)

            switch self {
            case .object(let obj):
                try container.encode("object", forKey: .type)
                if let desc = obj.description {
                    try container.encode(desc, forKey: .description)
                }
                var propsContainer = container.nestedContainer(keyedBy: DynamicCodingKey.self, forKey: .properties)
                for (name, node) in obj.properties {
                    try propsContainer.encode(node, forKey: DynamicCodingKey(stringValue: name)!)
                }
                try container.encode(Array(obj.required), forKey: .required)
                try container.encode(false, forKey: .additionalProperties)

            case .array(let arr):
                try container.encode("array", forKey: .type)
                if let desc = arr.description {
                    try container.encode(desc, forKey: .description)
                }
                try container.encode(arr.items, forKey: .items)
                if let min = arr.minItems {
                    try container.encode(min, forKey: .minItems)
                }
                if let max = arr.maxItems {
                    try container.encode(max, forKey: .maxItems)
                }

            case .string(let str):
                try container.encode("string", forKey: .type)
                if let desc = str.description {
                    try container.encode(desc, forKey: .description)
                }
                if let pattern = str.pattern {
                    try container.encode(pattern, forKey: .pattern)
                }
                if let choices = str.enumChoices {
                    try container.encode(choices, forKey: .enum)
                }

            case .number(let num):
                try container.encode(num.integerOnly ? "integer" : "number", forKey: .type)
                if let desc = num.description {
                    try container.encode(desc, forKey: .description)
                }
                if let min = num.minimum {
                    try container.encode(min, forKey: .minimum)
                }
                if let max = num.maximum {
                    try container.encode(max, forKey: .maximum)
                }

            case .boolean:
                try container.encode("boolean", forKey: .type)

            case .anyOf(let nodes):
                try container.encode(nodes, forKey: .anyOf)

            case .ref(let name):
                try container.encode("#/$defs/\(name)", forKey: .ref)
            }
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)

            if container.contains(.ref) {
                let refString = try container.decode(String.self, forKey: .ref)
                let name = refString.replacingOccurrences(of: "#/$defs/", with: "")
                self = .ref(name)
                return
            }

            if container.contains(.anyOf) {
                let nodes = try container.decode([Node].self, forKey: .anyOf)
                self = .anyOf(nodes)
                return
            }

            let type = try container.decode(String.self, forKey: .type)
            let description = try container.decodeIfPresent(String.self, forKey: .description)

            switch type {
            case "object":
                let propsContainer = try container.nestedContainer(keyedBy: DynamicCodingKey.self, forKey: .properties)
                var properties: [String: Node] = [:]
                for key in propsContainer.allKeys {
                    properties[key.stringValue] = try propsContainer.decode(Node.self, forKey: key)
                }
                let requiredArray = try container.decodeIfPresent([String].self, forKey: .required) ?? []
                let required = Set(requiredArray)
                self = .object(ObjectNode(description: description, properties: properties, required: required))

            case "array":
                let items = try container.decode(Node.self, forKey: .items)
                let minItems = try container.decodeIfPresent(Int.self, forKey: .minItems)
                let maxItems = try container.decodeIfPresent(Int.self, forKey: .maxItems)
                self = .array(ArrayNode(description: description, items: items, minItems: minItems, maxItems: maxItems))

            case "string":
                let pattern = try container.decodeIfPresent(String.self, forKey: .pattern)
                let enumChoices = try container.decodeIfPresent([String].self, forKey: .enum)
                self = .string(StringNode(description: description, pattern: pattern, enumChoices: enumChoices))

            case "number", "integer":
                let minimum = try container.decodeIfPresent(Double.self, forKey: .minimum)
                let maximum = try container.decodeIfPresent(Double.self, forKey: .maximum)
                self = .number(
                    NumberNode(
                        description: description,
                        minimum: minimum,
                        maximum: maximum,
                        integerOnly: type == "integer"
                    )
                )

            case "boolean":
                self = .boolean

            default:
                throw DecodingError.dataCorruptedError(
                    forKey: .type,
                    in: container,
                    debugDescription: "Unknown type: \(type)"
                )
            }
        }
    }

    struct ObjectNode: Sendable, Codable {
        var description: String?
        var properties: [String: Node]
        var required: Set<String>
    }

    struct ArrayNode: Sendable, Codable {
        var description: String?
        var items: Node
        var minItems: Int?
        var maxItems: Int?
    }

    struct StringNode: Sendable, Codable {
        var description: String?
        var pattern: String?
        var enumChoices: [String]?
    }

    struct NumberNode: Sendable, Codable {
        var description: String?
        var minimum: Double?
        var maximum: Double?
        var integerOnly: Bool
    }

    private struct DynamicCodingKey: CodingKey {
        var stringValue: String
        var intValue: Int?

        init?(stringValue: String) {
            self.stringValue = stringValue
            self.intValue = nil
        }

        init?(intValue: Int) {
            self.stringValue = String(intValue)
            self.intValue = intValue
        }
    }

    // MARK: - Properties

    let root: Node
    private var defs: [String: Node]

    /// A string representation of the debug description.
    ///
    /// This string is not localized and is not appropriate for display to end users.
    public var debugDescription: String {
        var parts: [String] = []
        parts.append("GenerationSchema:")
        parts.append("  root: \(debugString(for: root, indent: 2))")
        if !defs.isEmpty {
            parts.append("  $defs:")
            for (name, node) in defs.sorted(by: { $0.key < $1.key }) {
                parts.append("    \(name): \(debugString(for: node, indent: 4))")
            }
        }
        return parts.joined(separator: "\n")
    }

    private func debugString(for node: Node, indent: Int) -> String {
        switch node {
        case .object(let obj):
            return "object(\(obj.properties.count) properties)"
        case .array(let arr):
            return "array(items: \(debugString(for: arr.items, indent: 0)))"
        case .string(let str):
            if let choices = str.enumChoices {
                return "string(enum: \(choices))"
            } else if str.pattern != nil {
                return "string(pattern)"
            }
            return "string"
        case .number(let num):
            return num.integerOnly ? "integer" : "number"
        case .boolean:
            return "boolean"
        case .anyOf(let nodes):
            return "anyOf(\(nodes.count) choices)"
        case .ref(let name):
            return "$ref(\(name))"
        }
    }

    /// Creates a schema by providing an array of properties.
    ///
    /// - Parameters:
    ///   - type: The type this schema represents.
    ///   - description: A natural language description of this schema.
    ///   - properties: An array of properties.
    public init(
        type: any Generable.Type,
        description: String? = nil,
        properties: [GenerationSchema.Property]
    ) {
        let typeName = String(reflecting: type)
        var props: [String: Node] = [:]
        var required: Set<String> = []
        var allDefs: [String: Node] = [:]

        for property in properties {
            props[property.name] = property.node
            if !property.isOptional {
                required.insert(property.name)
            }
            for (defName, defNode) in property.deps {
                if let existing = allDefs[defName], !Self.nodesEqual(existing, defNode) {
                    fatalError("Duplicate type '\(defName)' with different structure")
                }
                allDefs[defName] = defNode
            }
        }

        let objectNode = ObjectNode(description: description, properties: props, required: required)
        allDefs[typeName] = .object(objectNode)

        self.root = .ref(typeName)
        self.defs = allDefs
    }

    /// Creates a schema for a string enumeration.
    ///
    /// - Parameters:
    ///   - type: The type this schema represents.
    ///   - description: A natural language description of this schema.
    ///   - anyOf: The allowed choices.
    public init(
        type: any Generable.Type,
        description: String? = nil,
        anyOf choices: [String]
    ) {
        guard !choices.isEmpty else {
            fatalError("Empty choices for enum schema")
        }
        let node = StringNode(description: description, pattern: nil, enumChoices: choices)
        self.root = .string(node)
        self.defs = [:]
    }

    /// Creates a schema as the union of several other types.
    ///
    /// - Parameters:
    ///   - type: The type this schema represents.
    ///   - description: A natural language description of this schema.
    ///   - anyOf: The types this schema should be a union of.
    public init(
        type: any Generable.Type,
        description: String? = nil,
        anyOf types: [any Generable.Type]
    ) {
        guard !types.isEmpty else {
            fatalError("Empty types for anyOf schema")
        }

        var members: [Node] = []
        var allDefs: [String: Node] = [:]

        for t in types {
            let tName = String(reflecting: t)
            members.append(.ref(tName))

            let tSchema = t.generationSchema
            for (defName, defNode) in tSchema.defs {
                if let existing = allDefs[defName], !Self.nodesEqual(existing, defNode) {
                    fatalError("Duplicate type '\(defName)' with different structure")
                }
                allDefs[defName] = defNode
            }

            if case .ref(_) = tSchema.root {
                // Already in defs
            } else {
                allDefs[tName] = tSchema.root
            }
        }

        self.root = .anyOf(members)
        self.defs = allDefs
    }

    /// Creates a schema by providing an array of dynamic schemas.
    ///
    /// - Parameters:
    ///   - root: The root schema.
    ///   - dependencies: An array of dynamic schemas.
    /// - Throws: Throws there are schemas with naming conflicts or
    ///   references to undefined types.
    public init(root: DynamicGenerationSchema, dependencies: [DynamicGenerationSchema]) throws {
        var nameMap: [String: DynamicGenerationSchema] = [:]
        var allDefs: [String: Node] = [:]

        // Build name map
        for dep in dependencies {
            if let name = dep.name {
                if nameMap[name] != nil {
                    throw SchemaError.duplicateType(
                        schema: nil,
                        type: name,
                        context: SchemaError.Context(debugDescription: "Duplicate dependency name")
                    )
                }
                nameMap[name] = dep
            }
        }

        if let rootName = root.name {
            if nameMap[rootName] != nil {
                throw SchemaError.duplicateType(
                    schema: nil,
                    type: rootName,
                    context: SchemaError.Context(debugDescription: "Root name conflicts with dependency")
                )
            }
            nameMap[rootName] = root
        }

        // Convert root
        let rootNode = try Self.convertDynamic(root, nameMap: nameMap, defs: &allDefs)

        // Convert all dependencies
        for dep in dependencies {
            _ = try Self.convertDynamic(dep, nameMap: nameMap, defs: &allDefs)
        }

        // Validate all references
        var undefinedRefs: [String] = []
        try Self.validateRefs(rootNode, defs: allDefs, undefinedRefs: &undefinedRefs)
        for (_, defNode) in allDefs {
            try Self.validateRefs(defNode, defs: allDefs, undefinedRefs: &undefinedRefs)
        }

        if !undefinedRefs.isEmpty {
            throw SchemaError.undefinedReferences(
                schema: root.name,
                references: Array(Set(undefinedRefs)),
                context: SchemaError.Context(debugDescription: "Undefined references")
            )
        }

        self.root = rootNode
        self.defs = allDefs
    }

    private static func convertDynamic(
        _ dynamic: DynamicGenerationSchema,
        nameMap: [String: DynamicGenerationSchema],
        defs: inout [String: Node]
    ) throws -> Node {
        switch dynamic.body {
        case .object(let name, let desc, let properties):
            var props: [String: Node] = [:]
            var required: Set<String> = []
            for prop in properties {
                props[prop.name] = try convertDynamic(prop.schema, nameMap: nameMap, defs: &defs)
                if !prop.isOptional {
                    required.insert(prop.name)
                }
            }
            let node = Node.object(ObjectNode(description: desc, properties: props, required: required))
            if let name = name {
                defs[name] = node
                return .ref(name)
            }
            return node

        case .anyOf(let name, _, let choices):
            let nodes = try choices.map { try convertDynamic($0, nameMap: nameMap, defs: &defs) }
            let node = Node.anyOf(nodes)
            if let name = name {
                defs[name] = node
                return .ref(name)
            }
            return node

        case .stringEnum(let name, let desc, let choices):
            guard !choices.isEmpty else {
                throw SchemaError.emptyTypeChoices(
                    schema: name ?? "",
                    context: SchemaError.Context(debugDescription: "Empty enum choices")
                )
            }
            let node = Node.string(StringNode(description: desc, pattern: nil, enumChoices: choices))
            if let name = name {
                defs[name] = node
                return .ref(name)
            }
            return node

        case .array(let item, let min, let max):
            let itemNode = try convertDynamic(item, nameMap: nameMap, defs: &defs)
            return .array(ArrayNode(description: nil, items: itemNode, minItems: min, maxItems: max))

        case .scalar(let scalar):
            switch scalar {
            case .bool:
                return .boolean
            case .string:
                return .string(StringNode(description: nil, pattern: nil, enumChoices: nil))
            case .number:
                return .number(NumberNode(description: nil, minimum: nil, maximum: nil, integerOnly: false))
            case .integer:
                return .number(NumberNode(description: nil, minimum: nil, maximum: nil, integerOnly: true))
            case .decimal:
                return .number(NumberNode(description: nil, minimum: nil, maximum: nil, integerOnly: false))
            }

        case .reference(let name):
            return .ref(name)
        }
    }

    private static func validateRefs(_ node: Node, defs: [String: Node], undefinedRefs: inout [String]) throws {
        switch node {
        case .ref(let name):
            if defs[name] == nil {
                undefinedRefs.append(name)
            }
        case .object(let obj):
            for (_, propNode) in obj.properties {
                try validateRefs(propNode, defs: defs, undefinedRefs: &undefinedRefs)
            }
        case .array(let arr):
            try validateRefs(arr.items, defs: defs, undefinedRefs: &undefinedRefs)
        case .anyOf(let nodes):
            for n in nodes {
                try validateRefs(n, defs: defs, undefinedRefs: &undefinedRefs)
            }
        default:
            break
        }
    }

    private static func nodesEqual(_ a: Node, _ b: Node) -> Bool {
        // Simple structural equality - could be enhanced
        switch (a, b) {
        case (.boolean, .boolean):
            return true
        case (.ref(let aName), .ref(let bName)):
            return aName == bName
        default:
            return false
        }
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        if container.contains(.defs) {
            let defsContainer = try container.nestedContainer(keyedBy: DynamicCodingKey.self, forKey: .defs)
            var defs: [String: Node] = [:]
            for key in defsContainer.allKeys {
                defs[key.stringValue] = try defsContainer.decode(Node.self, forKey: key)
            }
            self.defs = defs
        } else {
            self.defs = [:]
        }

        // Decode the root - could be inline or a ref
        if container.contains(.ref) {
            let refString = try container.decode(String.self, forKey: .ref)
            let name = refString.replacingOccurrences(of: "#/$defs/", with: "")
            self.root = .ref(name)
        } else {
            // Inline root - decode as a node
            self.root = try Node(from: decoder)
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)

        if !defs.isEmpty {
            var defsContainer = container.nestedContainer(keyedBy: DynamicCodingKey.self, forKey: .defs)
            for (name, node) in defs {
                try defsContainer.encode(node, forKey: DynamicCodingKey(stringValue: name)!)
            }
        }

        // Encode root
        if case .ref(_) = root {
            try root.encode(to: encoder)
        } else {
            try root.encode(to: encoder)
        }
    }

    private enum CodingKeys: String, CodingKey {
        case defs = "$defs"
        case ref = "$ref"
    }

    // MARK: - Helpers

    static func primitive<T: Generable>(_: T.Type, node: Node) -> GenerationSchema {
        GenerationSchema(root: node, defs: [:])
    }

    private init(root: Node, defs: [String: Node]) {
        self.root = root
        self.defs = defs
    }
}

// MARK: - GenerationSchema.Property

extension GenerationSchema {

    /// A property that belongs to a generation schema.
    ///
    /// Fields are named members of object types. Fields are strongly
    /// typed and have optional descriptions and guides.
    public struct Property: Sendable {
        let name: String
        let node: Node
        let isOptional: Bool
        var deps: [String: Node]

        /// Create a property that contains a generable type.
        ///
        /// - Parameters:
        ///   - name: The property's name.
        ///   - description: A natural language description of what content
        ///     should be generated for this property.
        ///   - type: The type this property represents.
        ///   - guides: A list of guides to apply to this property.
        public init<Value>(
            name: String,
            description: String? = nil,
            type: Value.Type,
            guides: [GenerationGuide<Value>] = []
        ) where Value: Generable {
            self.name = name
            self.isOptional = false

            let (node, deps) = Self.buildNode(for: Value.self, description: description, guides: guides)
            self.node = node
            self.deps = deps
        }

        /// Create an optional property that contains a generable type.
        ///
        /// - Parameters:
        ///   - name: The property's name.
        ///   - description: A natural language description of what content
        ///     should be generated for this property.
        ///   - type: The type this property represents.
        ///   - guides: A list of guides to apply to this property.
        public init<Value>(
            name: String,
            description: String? = nil,
            type: Value?.Type,
            guides: [GenerationGuide<Value>] = []
        ) where Value: Generable {
            self.name = name
            self.isOptional = true

            let (node, deps) = Self.buildNode(for: Value.self, description: description, guides: guides)
            self.node = node
            self.deps = deps
        }

        /// Create a property that contains a string type.
        ///
        /// - Parameters:
        ///   - name: The property's name.
        ///   - description: A natural language description of what content
        ///     should be generated for this property.
        ///   - type: The type this property represents.
        ///   - guides: An array of regexes to be applied to this string. If there're multiple regexes in the array, only the last one will be applied.
        public init<RegexOutput>(
            name: String,
            description: String? = nil,
            type: String.Type,
            guides: [Regex<RegexOutput>] = []
        ) {
            self.name = name
            self.isOptional = false
            self.node = .string(StringNode(description: description, pattern: nil, enumChoices: nil))
            self.deps = [:]
        }

        /// Create an optional property that contains a generable type.
        ///
        /// - Parameters:
        ///   - name: The property's name.
        ///   - description: A natural language description of what content
        ///     should be generated for this property.
        ///   - type: The type this property represents.
        ///   - guides: An array of regexes to be applied to this string. If there're multiple regexes in the array, only the last one will be applied.
        public init<RegexOutput>(
            name: String,
            description: String? = nil,
            type: String?.Type,
            guides: [Regex<RegexOutput>] = []
        ) {
            self.name = name
            self.isOptional = true
            self.node = .string(StringNode(description: description, pattern: nil, enumChoices: nil))
            self.deps = [:]
        }

        private static func buildNode<Value: Generable>(
            for type: Value.Type,
            description: String?,
            guides: [GenerationGuide<Value>]
        ) -> (Node, [String: Node]) {
            // Check if it's a primitive type
            if type == Bool.self {
                return (.boolean, [:])
            } else if type == String.self {
                return (.string(StringNode(description: description, pattern: nil, enumChoices: nil)), [:])
            } else if type == Int.self {
                return (
                    .number(NumberNode(description: description, minimum: nil, maximum: nil, integerOnly: true)), [:]
                )
            } else if type == Float.self || type == Double.self || type == Decimal.self {
                return (
                    .number(NumberNode(description: description, minimum: nil, maximum: nil, integerOnly: false)), [:]
                )
            } else {
                // Complex type - use its schema
                let schema = Value.generationSchema
                let typeName = String(reflecting: Value.self)

                var deps = schema.defs
                if case .ref(_) = schema.root {
                    // Already a ref
                } else {
                    deps[typeName] = schema.root
                }

                return (.ref(typeName), deps)
            }
        }
    }
}

// MARK: - GenerationSchema.SchemaError

extension GenerationSchema {

    /// A error that occurs when there is a problem creating a generation schema.
    public enum SchemaError: Error, LocalizedError {

        /// The context in which the error occurred.
        public struct Context: Sendable {

            /// A string representation of the debug description.
            ///
            /// This string is not localized and is not appropriate for display to end users.
            public let debugDescription: String

            public init(debugDescription: String) {
                self.debugDescription = debugDescription
            }
        }

        /// An error that represents an attempt to construct a schema from dynamic schemas,
        /// and two or more of the subschemas have the same type name.
        case duplicateType(schema: String?, type: String, context: Context)

        /// An error that represents an attempt to construct a dynamic schema
        /// with properties that have conflicting names.
        case duplicateProperty(schema: String, property: String, context: Context)

        /// An error that represents an attempt to construct an anyOf schema with an
        /// empty array of type choices.
        case emptyTypeChoices(schema: String, context: Context)

        /// An error that represents an attempt to construct a schema from dynamic schemas,
        /// and one of those schemas references an undefined schema.
        case undefinedReferences(schema: String?, references: [String], context: Context)

        /// A string representation of the error description.
        public var errorDescription: String? {
            switch self {
            case .duplicateType(let schema, let type, _):
                return "Duplicate type '\(type)' in schema '\(schema ?? "root")'"
            case .duplicateProperty(let schema, let property, _):
                return "Duplicate property '\(property)' in schema '\(schema)'"
            case .emptyTypeChoices(let schema, _):
                return "Empty type choices in schema '\(schema)'"
            case .undefinedReferences(let schema, let references, _):
                return "Undefined references \(references) in schema '\(schema ?? "root")'"
            }
        }

        /// A suggestion that indicates how to handle the error.
        public var recoverySuggestion: String? {
            switch self {
            case .duplicateType:
                return "Ensure all types have unique names"
            case .duplicateProperty:
                return "Ensure all properties have unique names"
            case .emptyTypeChoices:
                return "Provide at least one type choice"
            case .undefinedReferences:
                return "Ensure all referenced schemas are defined"
            }
        }
    }
}
