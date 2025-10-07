import struct Foundation.Decimal
import class Foundation.NSDecimalNumber

/// Guides that control how values are generated.
public struct GenerationGuide<Value> {}

// MARK: - String Guides

extension GenerationGuide where Value == String {

    /// Enforces that the string be precisely the given value.
    public static func constant(_ value: String) -> GenerationGuide<String> {
        GenerationGuide<String>()
    }

    /// Enforces that the string be one of the provided values.
    public static func anyOf(_ values: [String]) -> GenerationGuide<String> {
        GenerationGuide<String>()
    }

    /// Enforces that the string follows the pattern.
    public static func pattern<Output>(_ regex: Regex<Output>) -> GenerationGuide<String> {
        GenerationGuide<String>()
    }
}

// MARK: - Int Guides

extension GenerationGuide where Value == Int {

    /// Enforces a minimum value.
    ///
    /// Use a `minimum` generation guide --- whose bounds are inclusive --- to ensure the model produces
    /// a value greater than or equal to some minimum value. For example, you can specify that all characters
    /// in your game start at level 1:
    ///
    /// ```swift
    /// @Generable
    /// struct struct GameCharacter {
    ///     @Guide(description: "A creative name appropriate for a fantasy RPG character")
    ///     var name: String
    ///
    ///     @Guide(description: "A level for the character", .minimum(1))
    ///     var level: Int
    /// }
    /// ```
    public static func minimum(_ value: Int) -> GenerationGuide<Int> {
        GenerationGuide<Int>()
    }

    /// Enforces a maximum value.
    ///
    /// Use a `maximum` generation guide --- whose bounds are inclusive --- to ensure the model produces
    /// a value less than or equal to some maximum value. For example, you can specify that the highest level
    /// a character in your game can achieve is 100:
    ///
    /// ```swift
    /// @Generable
    /// struct struct GameCharacter {
    ///     @Guide(description: "A creative name appropriate for a fantasy RPG character")
    ///     var name: String
    ///
    ///     @Guide(description: "A level for the character", .maximum(100))
    ///     var level: Int
    /// }
    /// ```
    public static func maximum(_ value: Int) -> GenerationGuide<Int> {
        GenerationGuide<Int>()
    }

    /// Enforces values fall within a range.
    ///
    /// Use a `range` generation guide --- whose bounds are inclusive --- to ensure the model produces a
    /// value that falls within a range. For example, you can specify that the level of characters in your game
    /// are between 1 and 100:
    ///
    /// ```swift
    /// @Generable
    /// struct struct GameCharacter {
    ///     @Guide(description: "A creative name appropriate for a fantasy RPG character")
    ///     var name: String
    ///
    ///     @Guide(description: "A level for the character", .range(1...100))
    ///     var level: Int
    /// }
    /// ```
    public static func range(_ range: ClosedRange<Int>) -> GenerationGuide<Int> {
        GenerationGuide<Int>()
    }
}

// MARK: - Float Guides

extension GenerationGuide where Value == Float {

    /// Enforces a minimum value.
    ///
    /// The bounds are inclusive.
    public static func minimum(_ value: Float) -> GenerationGuide<Float> {
        GenerationGuide<Float>()
    }

    /// Enforces a maximum value.
    ///
    /// The bounds are inclusive.
    public static func maximum(_ value: Float) -> GenerationGuide<Float> {
        GenerationGuide<Float>()
    }

    /// Enforces values fall within a range.
    public static func range(_ range: ClosedRange<Float>) -> GenerationGuide<Float> {
        GenerationGuide<Float>()
    }
}

// MARK: - Decimal Guides

extension GenerationGuide where Value == Decimal {

    /// Enforces a minimum value.
    ///
    /// The bounds are inclusive.
    public static func minimum(_ value: Decimal) -> GenerationGuide<Decimal> {
        GenerationGuide<Decimal>()
    }

    /// Enforces a maximum value.
    ///
    /// The bounds are inclusive.
    public static func maximum(_ value: Decimal) -> GenerationGuide<Decimal> {
        GenerationGuide<Decimal>()
    }

    /// Enforces values fall within a range.
    public static func range(_ range: ClosedRange<Decimal>) -> GenerationGuide<Decimal> {
        GenerationGuide<Decimal>()
    }
}

// MARK: - Double Guides

extension GenerationGuide where Value == Double {

    /// Enforces a minimum value.
    /// The bounds are inclusive.
    public static func minimum(_ value: Double) -> GenerationGuide<Double> {
        GenerationGuide<Double>()
    }

    /// Enforces a maximum value.
    /// The bounds are inclusive.
    public static func maximum(_ value: Double) -> GenerationGuide<Double> {
        GenerationGuide<Double>()
    }

    /// Enforces values fall within a range.
    public static func range(_ range: ClosedRange<Double>) -> GenerationGuide<Double> {
        GenerationGuide<Double>()
    }
}

// MARK: - Array Guides

extension GenerationGuide {

    /// Enforces a minimum number of elements in the array.
    ///
    /// The bounds are inclusive.
    public static func minimumCount<Element>(_ count: Int) -> GenerationGuide<[Element]>
    where Value == [Element] {
        GenerationGuide<[Element]>()
    }

    /// Enforces a maximum number of elements in the array.
    ///
    /// The bounds are inclusive.
    public static func maximumCount<Element>(_ count: Int) -> GenerationGuide<[Element]>
    where Value == [Element] {
        GenerationGuide<[Element]>()
    }

    /// Enforces that the number of elements in the array fall within a closed range.
    public static func count<Element>(_ range: ClosedRange<Int>) -> GenerationGuide<[Element]>
    where Value == [Element] {
        GenerationGuide<[Element]>()
    }

    /// Enforces that the array has exactly a certain number elements.
    public static func count<Element>(_ count: Int) -> GenerationGuide<[Element]>
    where Value == [Element] {
        GenerationGuide<[Element]>()
    }

    /// Enforces a guide on the elements within the array.
    public static func element<Element>(_ guide: GenerationGuide<Element>) -> GenerationGuide<
        [Element]
    >
    where Value == [Element] {
        GenerationGuide<[Element]>()
    }
}

// MARK: - Never Array Guides

extension GenerationGuide where Value == [Never] {

    /// Enforces a minimum number of elements in the array.
    ///
    /// Bounds are inclusive.
    ///
    /// - Warning: This overload is only used for macro expansion. Don't call `GenerationGuide<[Never]>.minimumCount(_:)` on your own.
    public static func minimumCount(_ count: Int) -> GenerationGuide<Value> {
        GenerationGuide<Value>()
    }

    /// Enforces a maximum number of elements in the array.
    ///
    /// Bounds are inclusive.
    ///
    /// - Warning: This overload is only used for macro expansion. Don't call `GenerationGuide<[Never]>.maximumCount(_:)` on your own.
    public static func maximumCount(_ count: Int) -> GenerationGuide<Value> {
        GenerationGuide<Value>()
    }

    /// Enforces that the number of elements in the array fall within a closed range.
    ///
    /// - Warning: This overload is only used for macro expansion. Don't call `GenerationGuide<[Never]>.count(_:)` on your own.
    public static func count(_ range: ClosedRange<Int>) -> GenerationGuide<Value> {
        GenerationGuide<Value>()
    }

    /// Enforces that the array has exactly a certain number elements.
    ///
    /// - Warning: This overload is only used for macro expansion. Don't call `GenerationGuide<[Never]>.count(_:)` on your own.
    public static func count(_ count: Int) -> GenerationGuide<Value> {
        GenerationGuide<Value>()
    }
}
