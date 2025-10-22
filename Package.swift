// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import CompilerPluginSupport
import PackageDescription

let package = Package(
    name: "AnyLanguageModel",
    platforms: [
        .macOS(.v14),
        .macCatalyst(.v17),
        .iOS(.v17),
        .tvOS(.v17),
        .watchOS(.v10),
        .visionOS(.v1),
    ],

    products: [
        .library(
            name: "AnyLanguageModel",
            targets: ["AnyLanguageModel"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/swiftlang/swift-syntax.git", from: "600.0.0"),
        .package(url: "https://github.com/mattt/JSONSchema.git", from: "1.3.0"),
        .package(url: "https://github.com/mattt/EventSource.git", from: "1.2.0"),
        .package(url: "https://github.com/mattt/PartialJSONDecoder.git", from: "1.0.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift-examples/", branch: "main"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.0.0"),
        .package(url: "https://github.com/mattt/llama.swift", .upToNextMajor(from: "1.6818.0")),
    ],
    targets: [
        .target(
            name: "AnyLanguageModel",
            dependencies: [
                .target(name: "AnyLanguageModelMacros"),
                .product(name: "EventSource", package: "EventSource"),
                .product(name: "JSONSchema", package: "JSONSchema"),
                .product(name: "PartialJSONDecoder", package: "PartialJSONDecoder"),
                .product(name: "MLXLLM", package: "mlx-swift-examples"),
                .product(name: "MLXLMCommon", package: "mlx-swift-examples"),
                .product(name: "Transformers", package: "swift-transformers"),
                .product(name: "Llama", package: "llama.swift"),
            ]
        ),
        .macro(
            name: "AnyLanguageModelMacros",
            dependencies: [
                .product(name: "SwiftSyntaxMacros", package: "swift-syntax"),
                .product(name: "SwiftCompilerPlugin", package: "swift-syntax"),
            ]
        ),
        .testTarget(
            name: "AnyLanguageModelTests",
            dependencies: ["AnyLanguageModel"]
        ),
    ]
)
