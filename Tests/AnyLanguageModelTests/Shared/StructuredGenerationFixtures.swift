import Foundation

@testable import AnyLanguageModel

@Generable
enum Priority: Equatable {
    case low
    case medium
    case high
}

@Generable
struct SimpleString: Equatable {
    @Guide(description: "A greeting message")
    var message: String
}

@Generable
struct SimpleInt: Equatable {
    @Guide(description: "A count value", .minimum(0))
    var count: Int
}

@Generable
struct SimpleBool: Equatable {
    @Guide(description: "A boolean flag")
    var value: Bool
}

@Generable
struct SimpleDouble: Equatable {
    @Guide(description: "A temperature value")
    var temperature: Double
}

@Generable
struct OptionalFields: Equatable {
    @Guide(description: "A required name")
    var name: String

    @Guide(description: "An optional nickname")
    var nickname: String?
}

@Generable
struct BasicStruct: Equatable {
    @Guide(description: "Person's name")
    var name: String

    @Guide(description: "Person's age", .minimum(0))
    var age: Int

    @Guide(description: "Is the person active")
    var isActive: Bool

    @Guide(description: "Score value")
    var score: Double
}

@Generable
struct Address: Equatable {
    @Guide(description: "Street name")
    var street: String

    @Guide(description: "City name")
    var city: String

    @Guide(description: "Postal code")
    var postalCode: String
}

@Generable
struct ReusedNestedStruct: Equatable {
    @Guide(description: "Some text")
    var text: String
}

@Generable
struct ContainerWithDuplicateNestedType: Equatable {
    var first: ReusedNestedStruct
    var second: ReusedNestedStruct
}

@Generable
struct StructuredPerson: Equatable {
    @Guide(description: "Person's name")
    var name: String

    @Guide(description: "Person's age")
    var age: Int

    var address: Address
}

@Generable
struct TaskItem: Equatable {
    @Guide(description: "Task title")
    var title: String

    var priority: Priority

    @Guide(description: "Is completed")
    var isCompleted: Bool
}

@Generable
struct SimpleArray: Equatable {
    @Guide(description: "A list of color names")
    var colors: [String]
}

@Generable
struct MultiChoiceQuestion: Equatable {
    @Guide(description: "The quiz question")
    var text: String

    @Guide(.count(4))
    var choices: [String]

    var answer: String

    @Guide(description: "A brief explanation of why the answer is correct")
    var explanation: String
}
