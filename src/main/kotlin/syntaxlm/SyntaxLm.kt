package syntaxlm

import java.io.BufferedInputStream
import java.io.File
import java.io.FileInputStream
import java.io.InputStream

/** A source-preserving highlight span. Offsets are Kotlin String offsets (UTF-16 code units). */
data class HighlightSpan(
    val start: Int,
    val end: Int,
    val kind: String,
    val text: String,
)

private data class SourceToken(
    val text: String,
    val start: Int,
    val end: Int,
    val hint: String,
)

/** Loads and runs the matrix exported by syntaxlm.py. */
class SyntaxLmModel private constructor(
    private val weights: Map<String, FloatArray>,
) {
    companion object {
        private val MAGIC = byteArrayOf('S'.code.toByte(), 'Y'.code.toByte(), 'L'.code.toByte(), 'M'.code.toByte())
        private const val VERSION = 1
        private val KINDS = listOf(
            "plain", "keyword", "name", "function", "type", "builtin",
            "string", "number", "comment", "operator", "punctuation", "decorator",
        )

        fun load(file: File): SyntaxLmModel = FileInputStream(file).use { load(it) }

        fun load(input: InputStream): SyntaxLmModel {
            val reader = LittleEndianReader(BufferedInputStream(input))
            require(reader.readBytes(4).contentEquals(MAGIC)) { "not a syntaxlm matrix (expected SYLM)" }
            require(reader.readU16() == VERSION) { "unsupported syntaxlm matrix version" }
            val kindCount = reader.readU16()
            require(kindCount == KINDS.size) { "matrix has $kindCount kinds; runtime expects ${KINDS.size}" }
            val featureCount = reader.readU32()
            val kinds = List(kindCount) { reader.readString() }
            require(kinds == KINDS) { "matrix kind order does not match runtime" }

            val rows = LinkedHashMap<String, FloatArray>(featureCount)
            repeat(featureCount) {
                val feature = reader.readString()
                rows[feature] = FloatArray(kindCount) { reader.readFloat() }
            }
            return SyntaxLmModel(rows)
        }
    }

    fun highlight(source: String, language: String): List<HighlightSpan> {
        val tokens = tokenize(source, language)
        val spans = ArrayList<HighlightSpan>()
        var previousKind: String? = null
        for (index in tokens.indices) {
            val kind = predict(features(tokens, index, previousKind))
            previousKind = kind
            if (kind == "plain") continue
            val token = tokens[index]
            val span = HighlightSpan(token.start, token.end, kind, token.text)
            val previous = spans.lastOrNull()
            if (previous != null && previous.end == span.start && previous.kind == span.kind) {
                spans[spans.lastIndex] = previous.copy(end = span.end, text = source.substring(previous.start, span.end))
            } else {
                spans += span
            }
        }
        return spans
    }

    private fun predict(featureList: List<String>): String {
        val scores = FloatArray(KINDS.size)
        for (feature in featureList) {
            val row = weights[feature] ?: continue
            for (kind in KINDS.indices) scores[kind] += row[kind]
        }
        var best = 0
        for (index in 1 until KINDS.size) {
            if (scores[index] > scores[best]) best = index
        }
        return KINDS[best]
    }

    private fun features(tokens: List<SourceToken>, index: Int, previousKind: String?): List<String> {
        val token = tokens[index]
        val previous = if (index == 0) "<BOS>" else tokens[index - 1].text
        val next = if (index + 1 == tokens.size) "<EOS>" else tokens[index + 1].text
        return listOf(
            "bias",
            "text=${token.text}",
            "lower=${token.text.lowercase()}",
            "shape=${tokenShape(token.text)}",
            "hint=${token.hint}",
            "prev=$previous",
            "next=$next",
            "prev_kind=${previousKind ?: "<BOS>"}",
        )
    }

    private fun tokenize(source: String, languageInput: String): List<SourceToken> {
        val language = normalizeLanguage(languageInput)
        val result = ArrayList<SourceToken>()
        var index = 0
        while (index < source.length) {
            val start = index
            val char = source[index]
            if (char.isWhitespace()) {
                index++
                continue
            }
            if (source.startsWith("//", index) || char == '#') {
                index = source.indexOf('\n', index).let { if (it < 0) source.length else it }
                result += SourceToken(source.substring(start, index), start, index, "comment")
                continue
            }
            if (source.startsWith("/*", index)) {
                val close = source.indexOf("*/", index + 2)
                index = if (close < 0) source.length else close + 2
                result += SourceToken(source.substring(start, index), start, index, "comment")
                continue
            }

            var stringStart = start
            if (language == "python" && char in "rRuUbBfF" && index + 1 < source.length && source[index + 1] in "'\"`") {
                index++
                stringStart = start
            }
            val quote = source[index]
            if (quote in "'\"`") {
                index++
                while (index < source.length) {
                    if (source[index] == '\\') {
                        index = (index + 2).coerceAtMost(source.length)
                    } else if (source[index] == quote) {
                        index++
                        break
                    } else {
                        index++
                    }
                }
                result += SourceToken(source.substring(stringStart, index), stringStart, index, "string")
                continue
            }
            index = start

            if (char.isDigit() || (char == '.' && index + 1 < source.length && source[index + 1].isDigit())) {
                index++
                while (index < source.length && (source[index].isLetterOrDigit() || source[index] in "._+-")) {
                    val candidate = source[index]
                    if ((candidate == '+' || candidate == '-') && source[index - 1] !in "eE") break
                    index++
                }
                result += SourceToken(source.substring(start, index), start, index, "number")
                continue
            }
            if (isWordStart(char)) {
                index++
                while (index < source.length && isWordPart(source[index])) index++
                val text = source.substring(start, index)
                val next = source.substring(index).trimStart().firstOrNull()
                result += SourceToken(text, start, index, wordHint(text, language, next))
                continue
            }

            val operator = Data.OPERATORS.firstOrNull { source.startsWith(it, index) }
            if (operator != null) {
                index += operator.length
                result += SourceToken(operator, start, index, if (operator == "@") "decorator" else "operator")
                continue
            }
            if (char in "{}()[] .,;:") {
                index++
                if (char != ' ') result += SourceToken(char.toString(), start, index, "punctuation")
                continue
            }
            index++
            result += SourceToken(char.toString(), start, index, "plain")
        }
        return result
    }

    private fun wordHint(text: String, language: String, next: Char?): String {
        if (language == "json") return "plain"
        if (text in keywords(language)) return "keyword"
        if (language == "python" && (text in Data.PYTHON_BUILTINS || (text.startsWith("_") && text.endsWith("_")))) return "builtin"
        if (language != "python" && text in setOf("console", "Math", "JSON", "Promise", "require")) return "builtin"
        if (text.firstOrNull()?.isUpperCase() == true) return "type"
        return if (next == '(') "function" else "name"
    }

    private fun tokenShape(text: String): String {
        if (text.isNotEmpty() && text.all { it.isDigit() || it in ".xXabcdefABCDEF" }) return "number"
        if (text.matches(Regex("[A-Za-z_$][\\w$]*"))) {
            if (text.all { !it.isLetter() || it.isUpperCase() }) return "UPPER"
            if (text.first().isUpperCase()) return "Capitalized"
            if ('_' in text) return "snake"
            return "word"
        }
        if (text.startsWith("'") || text.startsWith("\"") || text.startsWith("`")) return "quoted"
        return text.map { if (it.isLetter()) 'a' else if (it.isDigit()) '0' else it }.joinToString("")
    }

    private fun normalizeLanguage(input: String): String = when (input.lowercase()) {
        "js", "jsx", "javascript", "ts", "tsx", "typescript" -> "javascript"
        "py", "python3" -> "python"
        "jsonc" -> "json"
        else -> input.lowercase()
    }

    private fun keywords(language: String): Set<String> = when (language) {
        "python" -> Data.PYTHON_KEYWORDS
        "javascript" -> Data.JAVASCRIPT_KEYWORDS
        "kotlin" -> Data.KOTLIN_KEYWORDS
        else -> emptySet()
    }

    private fun isWordStart(char: Char): Boolean = char == '_' || char == '$' || char.isLetter()
    private fun isWordPart(char: Char): Boolean = isWordStart(char) || char.isDigit()

    private class LittleEndianReader(private val input: InputStream) {
        fun readBytes(count: Int): ByteArray {
            val result = ByteArray(count)
            var offset = 0
            while (offset < count) {
                val read = input.read(result, offset, count - offset)
                require(read >= 0) { "truncated syntaxlm matrix" }
                offset += read
            }
            return result
        }

        fun readU16(): Int {
            val bytes = readBytes(2)
            return (bytes[0].toInt() and 0xff) or ((bytes[1].toInt() and 0xff) shl 8)
        }

        fun readU32(): Int {
            val bytes = readBytes(4)
            return (bytes[0].toInt() and 0xff) or
                ((bytes[1].toInt() and 0xff) shl 8) or
                ((bytes[2].toInt() and 0xff) shl 16) or
                ((bytes[3].toInt() and 0xff) shl 24)
        }

        fun readFloat(): Float = Float.fromBits(readU32())

        fun readString(): String = String(readBytes(readU16()), Charsets.UTF_8)
    }

    private object Data {
        val OPERATORS = listOf("===", "!==", "==", "!=", "=>", "<=", ">=", "&&", "||", "++", "--", "**", "//", "::", "+", "-", "*", "/", "%", "=", "<", ">", "!", "&", "|", "^", "~", "?", "@")
        val PYTHON_BUILTINS = setOf("abs", "all", "any", "bool", "bytes", "callable", "dict", "enumerate", "filter", "float", "format", "int", "isinstance", "iter", "len", "list", "map", "max", "min", "next", "object", "open", "print", "range", "repr", "reversed", "round", "set", "sorted", "str", "sum", "super", "tuple", "type", "zip")
        val PYTHON_KEYWORDS = setOf("False", "None", "True", "and", "as", "assert", "async", "await", "break", "case", "class", "continue", "def", "del", "elif", "else", "except", "finally", "for", "from", "global", "if", "import", "in", "is", "lambda", "match", "nonlocal", "not", "or", "pass", "raise", "return", "try", "while", "with", "yield")
        val JAVASCRIPT_KEYWORDS = setOf("as", "async", "await", "break", "case", "catch", "class", "const", "continue", "debugger", "default", "delete", "do", "else", "export", "extends", "finally", "for", "from", "function", "get", "if", "import", "in", "instanceof", "let", "new", "of", "return", "set", "static", "super", "switch", "this", "throw", "try", "typeof", "var", "void", "while", "with", "yield", "true", "false", "null", "undefined")
        val KOTLIN_KEYWORDS = setOf("abstract", "actual", "annotation", "as", "break", "by", "catch", "class", "companion", "const", "constructor", "continue", "data", "do", "else", "enum", "expect", "final", "finally", "for", "fun", "if", "import", "in", "infix", "inner", "interface", "internal", "is", "lateinit", "noinline", "null", "object", "open", "operator", "out", "override", "package", "private", "protected", "public", "reified", "return", "sealed", "setparam", "super", "suspend", "tailrec", "this", "throw", "try", "typealias", "typeof", "val", "var", "vararg", "when", "where", "while", "true", "false")
    }
}

/** Small application-facing provider; keep this class as the integration boundary in the Kotlin app. */
class SyntaxHighlightProvider(private val model: SyntaxLmModel) {
    constructor(matrix: InputStream) : this(SyntaxLmModel.load(matrix))
    constructor(matrixFile: File) : this(SyntaxLmModel.load(matrixFile))

    fun highlight(source: String, language: String): List<HighlightSpan> = model.highlight(source, language)
}

/** Optional command-line runner: kotlin SyntaxLmRunner matrix.bin python < source.py */
object SyntaxLmRunner {
    @JvmStatic
    fun main(args: Array<String>) {
        require(args.size in 2..3) { "usage: SyntaxLmRunner <matrix.bin> <language> [file]" }
        val source = if (args.size == 3) File(args[2]).readText() else System.`in`.bufferedReader().readText()
        val spans = SyntaxHighlightProvider(File(args[0])).highlight(source, args[1])
        for (span in spans) println("${span.start}\t${span.end}\t${span.kind}\t${span.text}")
    }
}
