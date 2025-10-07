import AnyLanguageModel

actor ToolSpy<T: Tool>: Tool where T.Arguments: Sendable, T.Output: Sendable {
    typealias Arguments = T.Arguments
    typealias Output = T.Output

    let base: T

    private(set) var calls: [(arguments: Arguments, result: Result<Output, Error>)] = []

    nonisolated var name: String { base.name }
    nonisolated var description: String { base.description }
    nonisolated var parameters: GenerationSchema { base.parameters }
    nonisolated var includesSchemaInInstructions: Bool { base.includesSchemaInInstructions }

    init(_ base: T) {
        self.base = base
    }

    func call(arguments: Arguments) async throws -> Output {
        do {
            let output = try await base.call(arguments: arguments)
            calls.append((arguments, .success(output)))
            return output
        } catch {
            calls.append((arguments, .failure(error)))
            throw error
        }
    }
}

func spy<T: Tool>(on tool: T) -> ToolSpy<T> where T.Arguments: Sendable, T.Output: Sendable {
    ToolSpy(tool)
}
