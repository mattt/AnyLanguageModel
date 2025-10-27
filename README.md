# AnyLanguageModel

A Swift package that provides an API-compatible, drop-in replacement for 
[Apple's Foundation Models framework](https://developer.apple.com/documentation/FoundationModels)
with support for custom language model providers.

## Features

### Supported Providers

- [x] Apple Foundation Models
- [x] Core ML models
- [x] Swift MLX models
- [x] llama.cpp (GGUF models)
- [x] Ollama [HTTP API](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [x] Anthropic [Messages API](https://docs.claude.com/en/api/messages)
- [x] OpenAI [Responses API](https://platform.openai.com/docs/api-reference/responses)

## Requirements

- Swift 6.1+
- iOS 17.0+ / macOS 14.0+ / visionOS 1.0+

## Installation

Add this package to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/mattt/AnyLanguageModel.git", branch: "main")
]
```

### Conditional Dependencies

AnyLanguageModel uses [Swift 6.1 traits](https://docs.swift.org/swiftpm/documentation/packagemanagerdocs/packagetraits/)
to conditionally include heavy dependencies,
allowing you to opt-in only to the language model backends you need.
This results in smaller binary sizes and faster build times.

**Available traits**:

- `CoreML`: Enables Core ML model support
  (depends on `huggingface/swift-transformers`)
- `MLX`: Enables MLX model support
  (depends on `ml-explore/mlx-swift-examples`)
- `Llama`: Enables llama.cpp support
  (requires `mattt/llama.swift`)

By default, no traits are enabled.
To enable specific traits, specify them in your package's dependencies:

```swift
// In your Package.swift
dependencies: [
    .package(
        url: "https://github.com/mattt/AnyLanguageModel.git",
        branch: "main",
        traits: ["CoreML", "MLX"] // Enable CoreML and MLX support
    )
]
```

## Usage

```swift
import AnyLanguageModel

// Core functionality (always available)
var models: [(any LanguageModel)] = [
    SystemLanguageModel(), // Apple Foundation Models
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

// Conditional models (require traits to be enabled)
#if CoreML
models.append(CoreMLLanguageModel(url: "path/to/some.mlmodelc")) // Compiled Core ML model
#endif

#if MLX
models.append(MLXLanguageModel(modelId: "mlx-community/Qwen3-0.6B-4bit"))
#endif

#if Llama
models.append(LlamaLanguageModel(modelPath: "/path/to/model.gguf"))
#endif

struct WeatherTool: Tool {
    let name = "getWeather"
    let description = "Retrieve the latest weather information for a city"

    @Generable
    struct Arguments {
        @Guide(description: "The city to fetch the weather for")
        var city: String
    }

    func call(arguments: Arguments) async throws -> String {
        "The weather in \(arguments.city) is sunny and 72°F / 23°C"
    }
}

for model in models {
    let session = LanguageModelSession(model: model, tools: [WeatherTool()])
    let response = try await session.respond(to: "What's the weather in Cupertino?")
    print(response.text) // "It's sunny and 72°F in Cupertino"
}
```

## Testing

Run the test suite to verify everything works correctly:

```bash
swift test
```

Tests for different language model backends have varying requirements:

- **CoreML tests**: `swift test --enable-trait CoreML` + `ENABLE_COREML_TESTS=1` + `HF_TOKEN` (downloads model from HuggingFace)
- **MLX tests**: `swift test --enable-trait MLX` + `ENABLE_MLX_TESTS=1` + `HF_TOKEN` (uses pre-defined model)
- **Llama tests**: `swift test --enable-trait Llama` + `LLAMA_MODEL_PATH` (points to local GGUF file)
- **Anthropic tests**: `ANTHROPIC_API_KEY` (no traits needed)
- **OpenAI tests**: `OPENAI_API_KEY` (no traits needed)
- **Ollama tests**: No setup needed (skips in CI)

Example setup for all backends:

```bash
# Environment variables
export ENABLE_COREML_TESTS=1
export ENABLE_MLX_TESTS=1
export HF_TOKEN=your_huggingface_token
export LLAMA_MODEL_PATH=/path/to/model.gguf
export ANTHROPIC_API_KEY=your_anthropic_key
export OPENAI_API_KEY=your_openai_key

# Run all tests with traits enabled
swift test --enable-trait CoreML --enable-trait MLX --enable-trait Llama
```
