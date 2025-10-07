/// A type that can be converted to generated content.
public protocol ConvertibleToGeneratedContent: InstructionsRepresentable, PromptRepresentable {
    /// An instance that represents the generated content.
    var generatedContent: GeneratedContent { get }
}

// MARK: - Default Implementations

extension ConvertibleToGeneratedContent {
    /// An instance that represents the instructions.
    public var instructionsRepresentation: Instructions {
        Instructions(generatedContent.jsonString)
    }

    /// An instance that represents a prompt.
    public var promptRepresentation: Prompt {
        Prompt(generatedContent.jsonString)
    }
}

// MARK: - Standard Library Extensions

extension Optional: ConvertibleToGeneratedContent, PromptRepresentable, InstructionsRepresentable
where Wrapped: ConvertibleToGeneratedContent {
    /// An instance that represents the generated content.
    public var generatedContent: GeneratedContent {
        switch self {
        case .none:
            return GeneratedContent(kind: .null)
        case .some(let value):
            return value.generatedContent
        }
    }
}

extension Array: ConvertibleToGeneratedContent where Element: ConvertibleToGeneratedContent {
    /// An instance that represents the generated content.
    public var generatedContent: GeneratedContent {
        GeneratedContent(elements: self.map { $0.generatedContent })
    }
}
