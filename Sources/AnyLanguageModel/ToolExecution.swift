/// A decision about how to handle a tool call.
public enum ToolExecutionDecision: Sendable {
    /// Execute the tool call using the associated tool.
    case execute

    /// Stop the session after tool calls are generated without executing them.
    case stop

    /// Provide tool output without executing the tool.
    ///
    /// Use this to supply results from an external system or cached responses.
    case provideOutput([Transcript.Segment])
}

/// A delegate that observes and controls tool execution for a session.
public protocol ToolExecutionDelegate: Sendable {
    /// Notifies the delegate when the model generates tool calls.
    ///
    /// - Parameters:
    ///   - toolCalls: The tool calls produced by the model.
    ///   - session: The session that generated the tool calls.
    func didGenerateToolCalls(_ toolCalls: [Transcript.ToolCall], in session: LanguageModelSession) async

    /// Asks the delegate how to handle a tool call.
    ///
    /// Return `.execute` to run the tool, `.stop` to halt after tool calls are generated,
    /// or `.provideOutput` to supply output without executing the tool.
    /// - Parameters:
    ///   - toolCall: The tool call to evaluate.
    ///   - session: The session requesting the decision.
    func toolCallDecision(for toolCall: Transcript.ToolCall, in session: LanguageModelSession) async
        -> ToolExecutionDecision

    /// Notifies the delegate after a tool call produces output.
    ///
    /// - Parameters:
    ///   - toolCall: The tool call that was handled.
    ///   - output: The output sent back to the model.
    ///   - session: The session that executed the tool call.
    func didExecuteToolCall(
        _ toolCall: Transcript.ToolCall,
        output: Transcript.ToolOutput,
        in session: LanguageModelSession
    ) async

    /// Notifies the delegate when a tool call fails.
    ///
    /// - Parameters:
    ///   - toolCall: The tool call that failed.
    ///   - error: The underlying error raised during execution.
    ///   - session: The session that attempted the tool call.
    func didFailToolCall(
        _ toolCall: Transcript.ToolCall,
        error: any Error,
        in session: LanguageModelSession
    ) async
}

// MARK: - Default Implementations

extension ToolExecutionDelegate {
    /// Provides a default no-op implementation.
    public func didGenerateToolCalls(_ toolCalls: [Transcript.ToolCall], in session: LanguageModelSession) async {}

    /// Provides a default decision that executes the tool call.
    public func toolCallDecision(
        for toolCall: Transcript.ToolCall,
        in session: LanguageModelSession
    ) async -> ToolExecutionDecision {
        .execute
    }

    /// Provides a default no-op implementation.
    public func didExecuteToolCall(
        _ toolCall: Transcript.ToolCall,
        output: Transcript.ToolOutput,
        in session: LanguageModelSession
    ) async {}

    /// Provides a default no-op implementation.
    public func didFailToolCall(
        _ toolCall: Transcript.ToolCall,
        error: any Error,
        in session: LanguageModelSession
    ) async {}
}
