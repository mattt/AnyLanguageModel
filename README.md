# AnyLanguageModel

A Swift package that provides a drop-in replacement for
[Apple's Foundation Models framework](https://developer.apple.com/documentation/FoundationModels)
with support for custom language model providers.
All you need to do is change your import statement:

```diff
- import FoundationModels
+ import AnyLanguageModel
```

```swift
import AnyLanguageModel

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

let model = SystemLanguageModel.default
let session = LanguageModelSession(model: model, tools: [WeatherTool()])

let response = try await session.respond {
    Prompt("How's the weather in Cupertino?")
}
print(response.content)
```

## Features

### Supported Providers

- [x] [Apple Foundation Models](https://developer.apple.com/documentation/FoundationModels)
- [x] [Core ML](https://developer.apple.com/documentation/coreml) models
- [x] [MLX](https://github.com/ml-explore/mlx-swift) models
- [x] [llama.cpp](https://github.com/ggml-org/llama.cpp) (GGUF models)
- [x] Ollama [HTTP API](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [x] Anthropic [Messages API](https://docs.claude.com/en/api/messages)
- [x] Google [Gemini API](https://ai.google.dev/api/generate-content)
- [x] OpenAI [Chat Completions API](https://platform.openai.com/docs/api-reference/chat)
- [x] OpenAI [Responses API](https://platform.openai.com/docs/api-reference/responses)

## Requirements

- Swift 6.1+
- iOS 17.0+ / macOS 14.0+ / visionOS 1.0+ / Linux

> [!IMPORTANT]
> A bug in Xcode 26 may cause build errors
> when targeting macOS 15 / iOS 18 or earlier
> (e.g. `Conformance of 'String' to 'Generable' is only available in macOS 26.0 or newer`).
> As a workaround, build your project with Xcode 16.
> For more information, see [issue #15](https://github.com/mattt/AnyLanguageModel/issues/15).

## Installation

Add this package to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/mattt/AnyLanguageModel.git", from: "0.3.0")
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
  (depends on `ml-explore/mlx-swift-lm`)
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

### Apple Foundation Models

Uses Apple's [system language model](https://developer.apple.com/documentation/FoundationModels)
(requires macOS 26 / iOS 26 / visionOS 26 or later).

```swift
let model = SystemLanguageModel.default
let session = LanguageModelSession(model: model)

let response = try await session.respond {
    Prompt("Explain quantum computing in one sentence")
}
```

> [!NOTE]  
> Image inputs are not yet supported by Apple Foundation Models.

### Core ML

Run [Core ML](https://developer.apple.com/documentation/coreml) models
(requires `CoreML` trait):

```swift
let model = CoreMLLanguageModel(url: URL(fileURLWithPath: "path/to/model.mlmodelc"))

let session = LanguageModelSession(model: model)
let response = try await session.respond {
    Prompt("Summarize this text")
}
```

Enable the trait in Package.swift:

```swift
.package(
    url: "https://github.com/mattt/AnyLanguageModel.git",
    branch: "main",
    traits: ["CoreML"]
)
```

> [!NOTE]  
> Image inputs are not currently supported with `CoreMLLanguageModel`.

### MLX

Run [MLX](https://github.com/ml-explore/mlx-swift) models on Apple Silicon
(requires `MLX` trait):

```swift
let model = MLXLanguageModel(modelId: "mlx-community/Qwen3-0.6B-4bit")

let session = LanguageModelSession(model: model)
let response = try await session.respond {
    Prompt("What is the capital of France?")
}
```

Vision support depends on the specific MLX model you load.
Use a vision‑capable model for multimodal prompts
(for example, a VLM variant).
The following shows extracting text from an image:

```swift
let ocr = try await session.respond(
    to: "Extract the total amount from this receipt",
    images: [
        .init(url: URL(fileURLWithPath: "/path/to/receipt_page1.png")),
        .init(url: URL(fileURLWithPath: "/path/to/receipt_page2.png"))
    ]
)
print(ocr.content)
```

Enable the trait in Package.swift:

```swift
.package(
    url: "https://github.com/mattt/AnyLanguageModel.git",
    branch: "main",
    traits: ["MLX"]
)
```

### llama.cpp (GGUF)

Run GGUF quantized models via [llama.cpp](https://github.com/ggml-org/llama.cpp)
(requires `Llama` trait):

```swift
let model = LlamaLanguageModel(modelPath: "/path/to/model.gguf")

let session = LanguageModelSession(model: model)
let response = try await session.respond {
    Prompt("Translate 'hello world' to Spanish")
}
```

Enable the trait in Package.swift:

```swift
.package(
    url: "https://github.com/mattt/AnyLanguageModel.git",
    branch: "main",
    traits: ["Llama"]
)
```

> [!NOTE]  
> Image inputs are not currently supported with `LlamaLanguageModel`.

### OpenAI

Supports both
[Chat Completions](https://platform.openai.com/docs/api-reference/chat) and
[Responses](https://platform.openai.com/docs/api-reference/responses) APIs:

```swift
let model = OpenAILanguageModel(
    apiKey: ProcessInfo.processInfo.environment["OPENAI_API_KEY"]!,
    model: "gpt-4o-mini"
)

let session = LanguageModelSession(model: model)
let response = try await session.respond(
    to: "List the objects you see",
    images: [
        .init(url: URL(string: "https://example.com/desk.jpg")!),
        .init(
            data: try Data(contentsOf: URL(fileURLWithPath: "/path/to/closeup.png")),
            mimeType: "image/png"
        )
    ]
)
print(response.content)
```

For OpenAI-compatible endpoints that use older Chat Completions API:

```swift
let model = OpenAILanguageModel(
    baseURL: URL(string: "https://api.example.com")!,
    apiKey: apiKey,
    model: "gpt-4o-mini",
    apiVariant: .chatCompletions
)
```

### Anthropic

Uses the [Messages API](https://docs.claude.com/en/api/messages) with Claude models:

```swift
let model = AnthropicLanguageModel(
    apiKey: ProcessInfo.processInfo.environment["ANTHROPIC_API_KEY"]!,
    model: "claude-sonnet-4-5-20250929"
)

let session = LanguageModelSession(model: model, tools: [WeatherTool()])
let response = try await session.respond {
    Prompt("What's the weather like in San Francisco?")
}
```

You can include images with your prompt.
You can point to remote URLs or construct from image data:

```swift
let response = try await session.respond(
    to: "Explain the key parts of this diagram",
    image: .init(
        data: try Data(contentsOf: URL(fileURLWithPath: "/path/to/diagram.png")),
        mimeType: "image/png"
    )
)
print(response.content)
```

### Google Gemini

Uses the [Gemini API](https://ai.google.dev/api/generate-content) with Gemini models:

```swift
let model = GeminiLanguageModel(
    apiKey: ProcessInfo.processInfo.environment["GEMINI_API_KEY"]!,
    model: "gemini-2.5-flash"
)

let session = LanguageModelSession(model: model, tools: [WeatherTool()])
let response = try await session.respond {
    Prompt("What's the weather like in Tokyo?")
}
```

Send images with your prompt using remote or local sources:

```swift
let response = try await session.respond(
    to: "Identify the plants in this photo",
    image: .init(url: URL(string: "https://example.com/garden.jpg")!)
)
print(response.content)
```

Gemini models use an internal ["thinking process"](https://ai.google.dev/gemini-api/docs/thinking)
that improves reasoning and multi-step planning.
You can configure how much Gemini should "think" using the `thinking` parameter:

```swift
// Enable thinking
var model = GeminiLanguageModel(
    apiKey: apiKey,
    model: "gemini-2.5-flash",
    thinking: true /* or `.dynamic` */,
)

// Set an explicit number of tokens for its thinking budget
model.thinking = .budget(1024)

// Revert to default configuration without thinking
model.thinking = false /* or `.disabled` */
```

Gemini supports [server-side tools](https://ai.google.dev/gemini-api/docs/google-search)
that execute transparently on Google's infrastructure:

```swift
let model = GeminiLanguageModel(
    apiKey: apiKey,
    model: "gemini-2.5-flash",
    serverTools: [
        .googleMaps(latitude: 35.6580, longitude: 139.7016) // Optional location
    ]
)
```

**Available server tools**:

- `.googleSearch`
  Grounds responses with real-time web information
- `.googleMaps`
  Provides location-aware responses
- `.codeExecution`
  Generates and runs Python code to solve problems
- `.urlContext`
  Fetches and analyzes content from URLs mentioned in prompts

> [!TIP]
> Gemini server tools are not available as client tools (`Tool`) for other models.

### Ollama

Run models locally via Ollama's
[HTTP API](https://github.com/ollama/ollama/blob/main/docs/api.md):

```swift
// Default: connects to http://localhost:11434
let model = OllamaLanguageModel(model: "qwen3") // `ollama pull qwen3:8b`

// Custom endpoint
let model = OllamaLanguageModel(
    endpoint: URL(string: "http://remote-server:11434")!,
    model: "llama3.2"
)

let session = LanguageModelSession(model: model)
let response = try await session.respond {
    Prompt("Tell me a joke")
}
```

For local models, make sure you’re using a vision‑capable model
(for example, a `-vl` variant).
You can combine multiple images:

```swift
let model = OllamaLanguageModel(model: "qwen3-vl") // `ollama pull qwen3-vl:8b`
let session = LanguageModelSession(model: model)
let response = try await session.respond(
    to: "Compare these posters and summarize their differences",
    images: [
        .init(url: URL(string: "https://example.com/poster1.jpg")!),
        .init(url: URL(fileURLWithPath: "/path/to/poster2.jpg"))
    ]
)
print(response.content)
```

## Testing

Run the test suite to verify everything works correctly:

```bash
swift test
```

Tests for different language model backends have varying requirements:

- **CoreML tests**: `swift test --traits CoreML` + `ENABLE_COREML_TESTS=1` + `HF_TOKEN` (downloads model from HuggingFace)
- **MLX tests**: `swift test --traits MLX` + `ENABLE_MLX_TESTS=1` + `HF_TOKEN` (uses pre-defined model)
- **Llama tests**: `swift test --traits Llama` + `LLAMA_MODEL_PATH` (points to local GGUF file)
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
swift test --traits CoreML,MLX,Llama
```
