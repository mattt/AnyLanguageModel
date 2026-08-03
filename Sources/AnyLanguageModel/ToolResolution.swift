/// A tool call paired with the output produced for it.
struct ToolInvocationResult: Sendable {
    let call: Transcript.ToolCall
    let output: Transcript.ToolOutput
}

/// The outcome of resolving a batch of model-generated tool calls.
enum ToolResolutionOutcome: Sendable {
    /// The session's delegate asked to stop before any of the calls ran.
    case stop(calls: [Transcript.ToolCall])

    /// The calls that were handled, along with their outputs.
    case invocations([ToolInvocationResult])
}

/// Executes model-generated tool calls, consulting the session's tool execution delegate.
///
/// Every model maps its provider-specific tool calls onto ``Transcript/ToolCall`` values and then
/// hands them here, so the delegate contract behaves identically no matter which model produced
/// the calls. See ``ToolExecutionDelegate`` for what a delegate can decide.
///
/// - Parameters:
///   - calls: The tool calls the model generated, in the order it produced them.
///   - session: The session whose tools and delegate handle the calls.
/// - Returns: ``ToolResolutionOutcome/stop(calls:)`` when the delegate halts the session, or
///   ``ToolResolutionOutcome/invocations(_:)`` with one result per call otherwise.
/// - Throws: ``LanguageModelSession/ToolCallError`` when a tool throws.
func resolveToolCalls(
    _ calls: [Transcript.ToolCall],
    session: LanguageModelSession
) async throws -> ToolResolutionOutcome {
    guard !calls.isEmpty else { return .invocations([]) }

    var toolsByName: [String: any Tool] = [:]
    for tool in session.tools where toolsByName[tool.name] == nil {
        toolsByName[tool.name] = tool
    }

    if let delegate = session.toolExecutionDelegate {
        await delegate.didGenerateToolCalls(calls, in: session)
    }

    var decisions: [ToolExecutionDecision] = []
    decisions.reserveCapacity(calls.count)

    if let delegate = session.toolExecutionDelegate {
        for call in calls {
            let decision = await delegate.toolCallDecision(for: call, in: session)
            if case .stop = decision {
                return .stop(calls: calls)
            }
            decisions.append(decision)
        }
    } else {
        decisions = Array(repeating: .execute, count: calls.count)
    }

    var results: [ToolInvocationResult] = []
    results.reserveCapacity(calls.count)

    for (index, call) in calls.enumerated() {
        switch decisions[index] {
        case .stop:
            // Unreachable: `.stop` returns while decisions are collected. Kept as a guard in case
            // that logic changes.
            return .stop(calls: calls)
        case .provideOutput(let segments):
            let output = Transcript.ToolOutput(
                id: call.id,
                toolName: call.toolName,
                segments: segments
            )
            if let delegate = session.toolExecutionDelegate {
                await delegate.didExecuteToolCall(call, output: output, in: session)
            }
            results.append(ToolInvocationResult(call: call, output: output))
        case .execute:
            guard let tool = toolsByName[call.toolName] else {
                let message = Transcript.Segment.text(.init(content: "Tool not found: \(call.toolName)"))
                let output = Transcript.ToolOutput(
                    id: call.id,
                    toolName: call.toolName,
                    segments: [message]
                )
                if let delegate = session.toolExecutionDelegate {
                    await delegate.didExecuteToolCall(call, output: output, in: session)
                }
                results.append(ToolInvocationResult(call: call, output: output))
                continue
            }

            do {
                let segments = try await tool.makeOutputSegments(from: call.arguments)
                let output = Transcript.ToolOutput(
                    id: call.id,
                    toolName: tool.name,
                    segments: segments
                )
                if let delegate = session.toolExecutionDelegate {
                    await delegate.didExecuteToolCall(call, output: output, in: session)
                }
                results.append(ToolInvocationResult(call: call, output: output))
            } catch {
                if let delegate = session.toolExecutionDelegate {
                    await delegate.didFailToolCall(call, error: error, in: session)
                }
                throw LanguageModelSession.ToolCallError(tool: tool, underlyingError: error)
            }
        }
    }

    return .invocations(results)
}
