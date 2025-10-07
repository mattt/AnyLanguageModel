import struct Foundation.Decimal

/// The dynamic counterpart to the generation schema type that you use to construct schemas at runtime.
///
/// An individual schema may reference other schemas by
/// name, and references are resolved when converting a set of
/// dynamic schemas into a ``GenerationSchema``.
public struct DynamicGenerationSchema: Sendable {

    internal indirect enum Body: Sendable {
        case object(name: String?, description: String?, properties: [Property])
        case anyOf(name: String?, description: String?, choices: [DynamicGenerationSchema])
        case stringEnum(name: String?, description: String?, choices: [String])
        case array(item: DynamicGenerationSchema, min: Int?, max: Int?)
        case scalar(Scalar)
        case reference(String)
    }

    internal enum Scalar: Sendable {
        case bool
        case string
        case number
        case integer
        case decimal
    }

    internal let body: Body
    internal var name: String? {
        switch body {
        case .object(let name, _, _), .anyOf(let name, _, _), .stringEnum(let name, _, _):
            return name
        case .array, .scalar, .reference:
            return nil
        }
    }

    /// Creates an object schema.
    ///
    /// - Parameters:
    ///   - name: A name this dynamic schema can be referenced by.
    ///   - description: A natural language description of this schema.
    ///   - properties: The properties to associated with this schema.
    public init(
        name: String,
        description: String? = nil,
        properties: [DynamicGenerationSchema.Property]
    ) {
        self.body = .object(name: name, description: description, properties: properties)
    }

    /// Creates an any-of schema.
    ///
    /// - Parameters:
    ///   - name: A name this schema can be referenecd by.
    ///   - description: A natural language description of this ``DynamicGenerationSchema``.
    ///   - choices: An array of schemas this one will be a union of.
    public init(
        name: String,
        description: String? = nil,
        anyOf choices: [DynamicGenerationSchema]
    ) {
        self.body = .anyOf(name: name, description: description, choices: choices)
    }

    /// Creates an enum schema.
    ///
    /// - Parameters:
    ///   - name: A name this schema can be referenced by.
    ///   - description: A natural language description of this ``DynamicGenerationSchema``.
    ///   - choices: An array of schemas this one will be a union of.
    public init(
        name: String,
        description: String? = nil,
        anyOf choices: [String]
    ) {
        self.body = .stringEnum(name: name, description: description, choices: choices)
    }

    /// Creates an array schema.
    ///
    /// - Parameters:
    ///   - arrayOf: A schema to use as the elements of the array.
    public init(
        arrayOf itemSchema: DynamicGenerationSchema,
        minimumElements: Int? = nil,
        maximumElements: Int? = nil
    ) {
        self.body = .array(item: itemSchema, min: minimumElements, max: maximumElements)
    }

    /// Creates a schema from a generable type and guides.
    ///
    /// - Parameters:
    ///   - type: A `Generable` type
    ///   - guides: Generation guides to apply to this `DynamicGenerationSchema`.
    public init<Value>(
        type: Value.Type,
        guides: [GenerationGuide<Value>] = []
    ) where Value: Generable {
        // Map to scalar types
        if type == Bool.self {
            self.body = .scalar(.bool)
        } else if type == String.self {
            self.body = .scalar(.string)
        } else if type == Int.self {
            self.body = .scalar(.integer)
        } else if type == Float.self || type == Double.self {
            self.body = .scalar(.number)
        } else if type == Decimal.self {
            self.body = .scalar(.decimal)
        } else {
            // Complex type - create a reference
            let typeName = String(reflecting: Value.self)
            self.body = .reference(typeName)
        }
    }

    /// Creates an refrence schema.
    ///
    /// - Parameters:
    ///   - name: The name of the ``DynamicGenerationSchema`` this is a reference to.
    public init(referenceTo name: String) {
        self.body = .reference(name)
    }
}

// MARK: - DynamicGenerationSchema.Property

extension DynamicGenerationSchema {
    /// A property that belongs to a dynamic generation schema.
    ///
    /// Fields are named members of object types. Fields are strongly
    /// typed and have optional descriptions.
    public struct Property: Sendable {
        internal let name: String
        internal let description: String?
        internal let schema: DynamicGenerationSchema
        internal let isOptional: Bool

        /// Creates a property referencing a dynamic schema.
        ///
        /// - Parameters:
        ///   - name: A name for this property.
        ///   - description: An optional natural language description of this
        ///     property's contents.
        ///   - schema: A schema representing the type this property contains.
        ///   - isOptional: Determines if this property is required or not.
        public init(
            name: String,
            description: String? = nil,
            schema: DynamicGenerationSchema,
            isOptional: Bool = false
        ) {
            self.name = name
            self.description = description
            self.schema = schema
            self.isOptional = isOptional
        }
    }
}
