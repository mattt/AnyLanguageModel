import Foundation

public protocol LanguageModel: Sendable {
    associatedtype UnavailableReason

    var availability: Availability<UnavailableReason> { get }

    var isResponding: Bool { get }

    func prewarm(
        for session: LanguageModelSession,
        promptPrefix: Prompt?
    )

    func respond<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<Content> where Content: Generable

    func streamResponse<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) -> sending LanguageModelSession.ResponseStream<Content> where Content: Generable

    func logFeedbackAttachment(
        within session: LanguageModelSession,
        sentiment: LanguageModelFeedback.Sentiment?,
        issues: [LanguageModelFeedback.Issue],
        desiredOutput: Transcript.Entry?
    ) -> Data
}

// MARK: - Default Implementation

extension LanguageModel {
    public var isAvailable: Bool {
        if case .available = availability {
            return true
        } else {
            return false
        }
    }

    public var isResponding: Bool {
        return false
    }

    public func prewarm(
        for session: LanguageModelSession,
        promptPrefix: Prompt? = nil
    ) {
        return
    }

    public func logFeedbackAttachment(
        within session: LanguageModelSession,
        sentiment: LanguageModelFeedback.Sentiment? = nil,
        issues: [LanguageModelFeedback.Issue] = [],
        desiredOutput: Transcript.Entry? = nil
    ) -> Data {
        return Data()
    }
}

extension LanguageModel where UnavailableReason == Never {
    public var availability: Availability<UnavailableReason> {
        return .available
    }
}
