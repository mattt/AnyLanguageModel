/// A prompt from a person to the model.
///
/// Prompts can contain content written by you, an outside source, or input directly from people using
/// your app. You can initialize a `Prompt` from a string literal:
///
/// ```swift
/// let prompt = Prompt("What are miniature schnauzers known for?")
/// ```
///
/// Use ``PromptBuilder`` to dynamically control the prompt's content based on your app's state. The
/// code below shows if the Boolean is `true`, the prompt includes a second line of text:
///
/// ```swift
/// let responseShouldRhyme = true
/// let prompt = Prompt {
///     "Answer the following question from the user: \(userInput)"
///     if responseShouldRhyme {
///         "Your response MUST rhyme!"
///     }
/// }
/// ```
///
/// If your prompt includes input from people, consider wrapping the input in a string template with your
/// own prompt to better steer the model's response. For more information on handling inputs in your
/// prompts, see <doc:improving-safety-from-generative-model-output>.
public struct Prompt: Sendable {
    private let content: String

    /// Creates an instance with the content you specify.
    public init(_ representable: some PromptRepresentable) {
        switch representable {
        case let prompt as Prompt:
            self = prompt
        case let string as String:
            self.init(content: string)
        default:
            self.init(content: representable.promptRepresentation.content)
        }
    }

    init(content: String) {
        self.content = content
    }

    public init(@PromptBuilder _ content: () throws -> Prompt) rethrows {
        self = try content()
    }
}

// MARK: - CustomStringConvertible

extension Prompt: CustomStringConvertible {
    public var description: String { content }
}

// MARK: - PromptBuilder

@resultBuilder
public struct PromptBuilder {
    public static func buildBlock<each P>(_ components: repeat each P) -> Prompt
    where repeat each P: PromptRepresentable {
        var parts: [String] = []
        repeat parts.append((each components).promptRepresentation.description)
        let combinedText = parts.joined(separator: "\n")
        return Prompt(content: combinedText.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    public static func buildExpression<P>(_ expression: P) -> P where P: PromptRepresentable {
        return expression
    }

    public static func buildArray(_ prompts: [some PromptRepresentable]) -> Prompt {
        let combinedText = prompts.map {
            $0.promptRepresentation.description
        }.joined(separator: "\n")
        return Prompt(content: combinedText)
    }

    public static func buildOptional(_ component: Prompt?) -> Prompt {
        return component ?? Prompt(content: "")
    }

    public static func buildEither(first component: some PromptRepresentable) -> Prompt {
        return component.promptRepresentation
    }

    public static func buildEither(second component: some PromptRepresentable) -> Prompt {
        return component.promptRepresentation
    }

    public static func buildLimitedAvailability(_ prompt: some PromptRepresentable) -> Prompt {
        return prompt.promptRepresentation
    }
}

// MARK: - PromptRepresentable

/// A protocol that represents a prompt.
public protocol PromptRepresentable {
    /// An instance that represents a prompt.
    var promptRepresentation: Prompt { get }
}

// MARK: - Default Implementations

extension Prompt: PromptRepresentable {
    /// An instance that represents a prompt.
    public var promptRepresentation: Prompt { self }
}

// MARK: - Standard Library Extensions

extension String: PromptRepresentable {
    /// An instance that represents a prompt.
    public var promptRepresentation: Prompt {
        Prompt(content: self)
    }
}

extension Array: PromptRepresentable where Element: PromptRepresentable {
    /// An instance that represents a prompt.
    public var promptRepresentation: Prompt {
        let combined = self.map { $0.promptRepresentation.description }
            .joined(separator: "\n")
        return Prompt(content: combined)
    }
}
