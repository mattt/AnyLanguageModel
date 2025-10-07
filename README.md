# AnyLanguageModel

A Swift package providing an API-compatible drop-in replacement adapter for Apple's Foundation Models framework.
This lets developers use `LanguageModelSession` APIs with models other than system-provided ones.

> [!WARNING]
> This package is under active development and may be unstable. Use at your own risk.

## Features

### Supported Providers

- [x] Apple Foundation Models
- [ ] Swift Transformers models
- [ ] Swift MLX models
- [x] Ollama [HTTP API](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [x] Anthropic [Messages API](https://docs.claude.com/en/api/messages)
- [x] OpenAI [Responses API](https://platform.openai.com/docs/api-reference/responses)
- [ ] Compound provider with fallbacks
- [x] Mocked responses

## Requirements

- Swift 6.0+
- iOS 17.0+ / macOS 14.0+ / visionOS 1.0+

## Installation

Add this package to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/mattt/AnyLanguageModel.git", branch: "main")
]
```

## Usage

```swift
import AnyLanguageModel
import Foundation

import MLX
import CoreML

let models = [(any LanguageModel)] = [
    SystemLanguageModel(), // Apple Foundation Models
    CoreMLModel(model: URL(filePath: "model.mlpackage"), computeUnits: .cpuAndNeuralEngine),
    MLXModel(model: try await loadModel(id: "mlx-community/Qwen3-4B-4bit"))
    OllamaLanguageModel(model: "qwen3") // `ollama pull qwen3:0.6b`
    AnthropicLanguageModel(
        apiKey: ProcessInfo.processInfo.environment["ANTHROPIC_API_KEY"]!,
        model: "claude-sonnet-4-5-20250929"
    ),
    OpenAILanguageModel(
        apiKey: ProcessInfo.processInfo.environment["OPENAI_API_KEY"]!,
        model: "gpt-4o-mini"
    ),
]

for model in models {
    let session = LanguageModelSession(model: model, tools: [WeatherTool()])
    let response = try await session.respond(to: "What's the weather in Cupertino?")
    let response = try await session.respond(to: Prompt("Say 'Hello'"))
    print(response.text) // "Hello"
}
```
