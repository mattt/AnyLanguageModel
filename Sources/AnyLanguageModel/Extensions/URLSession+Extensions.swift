import EventSource
import Foundation
import JSONSchema

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

package enum HTTP {
    package enum Method: String {
        case get = "GET"
        case post = "POST"
    }
}

extension URLSession {
    package func fetch<T: Decodable>(
        _ method: HTTP.Method,
        url: URL,
        headers: [String: String] = [:],
        body: Data? = nil,
        dateDecodingStrategy: JSONDecoder.DateDecodingStrategy = .deferredToDate
    ) async throws -> T {
        var request = URLRequest(url: url)
        request.httpMethod = method.rawValue
        request.addValue("application/json", forHTTPHeaderField: "Accept")

        for (key, value) in headers {
            request.addValue(value, forHTTPHeaderField: key)
        }

        if let body {
            request.httpBody = body
            request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        }

        let (data, response) = try await data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw URLSessionError.invalidResponse
        }

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = dateDecodingStrategy

        guard (200 ..< 300).contains(httpResponse.statusCode) else {
            if let errorString = String(data: data, encoding: .utf8) {
                throw URLSessionError.httpError(statusCode: httpResponse.statusCode, detail: errorString)
            }
            throw URLSessionError.httpError(statusCode: httpResponse.statusCode, detail: "Invalid response")
        }

        do {
            return try decoder.decode(T.self, from: data)
        } catch {
            throw URLSessionError.decodingError(detail: error.localizedDescription)
        }
    }

    package func fetchStream<T: Decodable & Sendable>(
        _ method: HTTP.Method,
        url: URL,
        headers: [String: String] = [:],
        body: Data? = nil,
        dateDecodingStrategy: JSONDecoder.DateDecodingStrategy = .deferredToDate
    ) -> AsyncThrowingStream<T, any Error> {
        AsyncThrowingStream { continuation in
            let task = Task { @Sendable in
                let decoder = JSONDecoder()
                decoder.dateDecodingStrategy = dateDecodingStrategy

                do {
                    var request = URLRequest(url: url)
                    request.httpMethod = method.rawValue
                    request.addValue("application/json", forHTTPHeaderField: "Accept")

                    for (key, value) in headers {
                        request.addValue(value, forHTTPHeaderField: key)
                    }

                    if let body {
                        request.httpBody = body
                        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
                    }

                    let (bytes, response) = try await bytes(for: request)

                    guard let httpResponse = response as? HTTPURLResponse else {
                        throw URLSessionError.invalidResponse
                    }

                    guard (200 ..< 300).contains(httpResponse.statusCode) else {
                        var errorData = Data()
                        for try await byte in bytes {
                            errorData.append(byte)
                        }

                        if let errorString = String(data: errorData, encoding: .utf8) {
                            throw URLSessionError.httpError(statusCode: httpResponse.statusCode, detail: errorString)
                        }
                        throw URLSessionError.httpError(statusCode: httpResponse.statusCode, detail: "Invalid response")
                    }

                    var buffer = Data()

                    for try await byte in bytes {
                        buffer.append(byte)

                        while let newlineIndex = buffer.firstIndex(of: UInt8(ascii: "\n")) {
                            let chunk = buffer[..<newlineIndex]
                            buffer = buffer[buffer.index(after: newlineIndex)...]

                            if !chunk.isEmpty {
                                let decoded = try decoder.decode(T.self, from: chunk)
                                continuation.yield(decoded)
                            }
                        }
                    }

                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }

    package func fetchEventStream<T: Decodable & Sendable>(
        _ method: HTTP.Method,
        url: URL,
        headers: [String: String] = [:],
        body: Data? = nil
    ) -> AsyncThrowingStream<T, any Error> {
        AsyncThrowingStream { continuation in
            let task = Task { @Sendable in
                do {
                    var request = URLRequest(url: url)
                    request.httpMethod = method.rawValue
                    request.addValue("text/event-stream", forHTTPHeaderField: "Accept")

                    for (key, value) in headers {
                        request.addValue(value, forHTTPHeaderField: key)
                    }

                    if let body {
                        request.httpBody = body
                        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
                    }

                    let (bytes, _) = try await bytes(for: request)
                    let decoder = JSONDecoder()

                    for try await event in bytes.events {
                        guard let data = event.data.data(using: .utf8) else { continue }

                        if let decoded = try? decoder.decode(T.self, from: data) {
                            continuation.yield(decoded)
                        }
                    }

                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }
}

enum URLSessionError: Error, CustomStringConvertible {
    case invalidResponse
    case httpError(statusCode: Int, detail: String)
    case decodingError(detail: String)

    var description: String {
        switch self {
        case .invalidResponse:
            return "Invalid response"
        case .httpError(let statusCode, let detail):
            return "HTTP error (Status \(statusCode)): \(detail)"
        case .decodingError(let detail):
            return "Decoding error: \(detail)"
        }
    }
}
