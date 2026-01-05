import struct Foundation.Decimal
import class Foundation.NSDecimalNumber

/// A type that the model uses when responding to prompts.
///
/// Annotate your Swift structure or enumeration with the `@Generable` macro to allow the model to
/// respond to prompts by generating an instance of your type. Use the `@Guide` macro to provide natural
/// language descriptions of your properties, and programmatically control the values that the model can
/// generate.
///
/// ```swift
/// @Generable
/// struct SearchSuggestions {
///     @Guide(description: "A list of suggested search terms", .count(4))
///     var searchTerms: [SearchTerm]
///
///     @Generable
///     struct SearchTerm {
///         // Use a generation identifier for data structures the framework generates.
///         var id: GenerationID
///
///         @Guide(description: "A 2 or 3 word search term, like 'Beautiful sunsets'")
///         var searchTerm: String
///     }
/// }
/// ```
public protocol Generable: ConvertibleFromGeneratedContent, ConvertibleToGeneratedContent {
    /// A representation of partially generated content
    associatedtype PartiallyGenerated: ConvertibleFromGeneratedContent = Self

    /// An instance of the generation schema.
    static var generationSchema: GenerationSchema { get }
}

// MARK: - Error

public enum GeneratedContentConversionError: Error {
    case typeMismatch
    case neverCannotBeInstantiated
}

// MARK: - Macros

/// Conforms a type to ``Generable`` protocol.
///
/// This macro synthesizes a memberwise initializer
/// and an `init(_ generatedContent: GeneratedContent)` initializer
/// for the annotated type.
///
/// - Note: The synthesized memberwise initializer isn't visible inside other macro bodies,
///         such as `#Playground`.
///         As a workaround, use the `init(_ generatedContent:)` initializer
///         or define a factory method on the type.
@attached(extension, conformances: Generable, names: named(init(_:)), named(generatedContent))
@attached(member, names: arbitrary)
public macro Generable(description: String? = nil) =
    #externalMacro(module: "AnyLanguageModelMacros", type: "GenerableMacro")

/// Allows for influencing the allowed values of properties of a generable type.
@attached(peer)
public macro Guide(description: String) =
    #externalMacro(module: "AnyLanguageModelMacros", type: "GuideMacro")

/// Allows for influencing the allowed values of properties of a generable type.
@attached(peer)
public macro Guide<T>(description: String? = nil, _ guides: GenerationGuide<T>...) =
    #externalMacro(module: "AnyLanguageModelMacros", type: "GuideMacro") where T: Generable

/// Allows for influencing the allowed values of properties of a generable type.
@attached(peer)
public macro Guide<RegexOutput>(
    description: String? = nil,
    _ guides: Regex<RegexOutput>
) = #externalMacro(module: "AnyLanguageModelMacros", type: "GuideMacro")

// MARK: - Default Implementations

extension Generable {
    /// The partially generated type of this struct.
    public func asPartiallyGenerated() -> Self.PartiallyGenerated {
        self as! Self.PartiallyGenerated
    }
}

extension Generable {
    /// A representation of partially generated content
    public typealias PartiallyGenerated = Self
}

// MARK: - Standard Library Extensions

// MARK: Optional

extension Optional where Wrapped: Generable {
    public typealias PartiallyGenerated = Wrapped.PartiallyGenerated
}

// MARK: Bool

extension Bool: Generable {

    /// An instance of the generation schema.
    public static var generationSchema: GenerationSchema {
        // Bool is a primitive, create a simple schema
        GenerationSchema.primitive(Bool.self, node: .boolean)
    }

    /// Creates an instance with the content.
    public init(_ content: GeneratedContent) throws {
        guard case .bool(let value) = content.kind else {
            throw GeneratedContentConversionError.typeMismatch
        }
        self = value
    }

    /// An instance that represents the generated content.
    public var generatedContent: GeneratedContent {
        GeneratedContent(kind: .bool(self))
    }
}

// MARK: String

extension String: Generable {
    /// An instance of the generation schema.
    public static var generationSchema: GenerationSchema {
        GenerationSchema.primitive(
            String.self,
            node: .string(GenerationSchema.StringNode(description: nil, pattern: nil, enumChoices: nil))
        )
    }

    /// Creates an instance with the content.
    public init(_ content: GeneratedContent) throws {
        guard case .string(let value) = content.kind else {
            throw GeneratedContentConversionError.typeMismatch
        }
        self = value
    }

    /// An instance that represents the generated content.
    public var generatedContent: GeneratedContent {
        GeneratedContent(kind: .string(self))
    }
}

// MARK: Int

extension Int: Generable {
    /// An instance of the generation schema.
    public static var generationSchema: GenerationSchema {
        GenerationSchema.primitive(
            Int.self,
            node: .number(GenerationSchema.NumberNode(description: nil, minimum: nil, maximum: nil, integerOnly: true))
        )
    }

    /// Creates an instance with the content.
    public init(_ content: GeneratedContent) throws {
        guard case .number(let value) = content.kind else {
            throw GeneratedContentConversionError.typeMismatch
        }
        self = Int(value)
    }

    /// An instance that represents the generated content.
    public var generatedContent: GeneratedContent {
        GeneratedContent(kind: .number(Double(self)))
    }
}

// MARK: Float

extension Float: Generable {
    /// An instance of the generation schema.
    public static var generationSchema: GenerationSchema {
        GenerationSchema.primitive(
            Float.self,
            node: .number(GenerationSchema.NumberNode(description: nil, minimum: nil, maximum: nil, integerOnly: false))
        )
    }

    /// Creates an instance with the content.
    public init(_ content: GeneratedContent) throws {
        guard case .number(let value) = content.kind else {
            throw GeneratedContentConversionError.typeMismatch
        }
        self = Float(value)
    }

    /// An instance that represents the generated content.
    public var generatedContent: GeneratedContent {
        GeneratedContent(kind: .number(Double(self)))
    }
}

// MARK: Double

extension Double: Generable {
    /// An instance of the generation schema.
    public static var generationSchema: GenerationSchema {
        GenerationSchema.primitive(
            Double.self,
            node: .number(GenerationSchema.NumberNode(description: nil, minimum: nil, maximum: nil, integerOnly: false))
        )
    }

    /// Creates an instance with the content.
    public init(_ content: GeneratedContent) throws {
        guard case .number(let value) = content.kind else {
            throw GeneratedContentConversionError.typeMismatch
        }
        self = value
    }

    /// An instance that represents the generated content.
    public var generatedContent: GeneratedContent {
        GeneratedContent(kind: .number(self))
    }
}

// MARK: Decimal

extension Decimal: Generable {
    /// An instance of the generation schema.
    public static var generationSchema: GenerationSchema {
        GenerationSchema.primitive(
            Decimal.self,
            node: .number(GenerationSchema.NumberNode(description: nil, minimum: nil, maximum: nil, integerOnly: false))
        )
    }

    /// Creates an instance with the content.
    public init(_ content: GeneratedContent) throws {
        guard case .number(let value) = content.kind else {
            throw GeneratedContentConversionError.typeMismatch
        }
        self = Decimal(value)
    }

    /// An instance that represents the generated content.
    public var generatedContent: GeneratedContent {
        let doubleValue = (self as NSDecimalNumber).doubleValue
        return GeneratedContent(kind: .number(doubleValue))
    }
}

// MARK: Array

extension Array: Generable where Element: Generable {
    /// A representation of partially generated content
    public typealias PartiallyGenerated = [Element.PartiallyGenerated]

    /// An instance of the generation schema.
    public static var generationSchema: GenerationSchema {
        let elementSchema = Element.generationSchema
        let arrayNode = GenerationSchema.ArrayNode(
            description: nil,
            items: elementSchema.root,
            minItems: nil,
            maxItems: nil
        )
        return GenerationSchema.primitive(
            [Element].self,
            node: .array(arrayNode)
        )
    }
}

// MARK: Never

extension Never: Generable {
    /// An instance of the generation schema.
    public static var generationSchema: GenerationSchema {
        fatalError("Never cannot be instantiated")
    }

    /// Creates an instance with the content.
    public init(_ content: GeneratedContent) throws {
        throw GeneratedContentConversionError.neverCannotBeInstantiated
    }

    /// An instance that represents the generated content.
    public var generatedContent: GeneratedContent {
        fatalError("Never cannot be instantiated")
    }
}
