extension Character {
    package static let jsonQuoteScalars: Set<UInt32> = [0x22, 0x201C, 0x201D, 0x2018, 0x2019]
    package static let jsonAllowedWhitespaceCharacters: Set<Character> = [" ", "\t", "\n"]

    package var containsEmojiScalar: Bool {
        unicodeScalars.contains { scalar in
            scalar.properties.isEmojiPresentation || scalar.properties.isEmoji
        }
    }

    package var isValidJSONStringCharacter: Bool {
        guard self != "\\" else { return false }
        guard let scalar = unicodeScalars.first, scalar.value >= 0x20 else { return false }
        guard !Self.jsonQuoteScalars.contains(scalar.value) else { return false }

        if let ascii = asciiValue {
            let char = Character(UnicodeScalar(ascii))
            if Self.jsonAllowedWhitespaceCharacters.contains(char) { return true }
            return isLetter || isNumber || (isASCII && (isPunctuation || isSymbol))
        }

        // Allow non-ASCII letters/numbers and emoji, but disallow non-ASCII punctuation (e.g. "】")
        return isLetter || isNumber || containsEmojiScalar
    }
}
