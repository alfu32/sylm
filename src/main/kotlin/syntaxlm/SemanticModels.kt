package syntaxlm

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.tanh

/** All positions are inclusive start/exclusive end UTF-16 offsets. */
data class SymbolResult(
    val id: String, val documentId: String, val revision: String,
    val start: Int, val end: Int, val name: String, val kind: Int, val confidence: Float,
    val contextEnd: Int? = null,
) {
    init {
        require(id.isNotEmpty() && documentId.isNotEmpty() && revision.isNotEmpty())
        require(start >= 0 && end >= start && kind in 0..7 && confidence in 0f..1f)
        require(contextEnd == null || contextEnd >= end)
    }
}

data class SemanticSnapshot(
    val documentId: String, val revision: String,
    val definitions: List<SymbolResult>, val usages: List<SymbolResult>,
    val definitionLanguages: FloatArray, val usageLanguages: FloatArray,
)

data class DefinitionResolution(
    val usage: SymbolResult, val definition: SymbolResult?, val confidence: Float,
    val candidateProbabilities: FloatArray,
) { val status: String get() = if (definition == null) "unresolved" else "resolved" }

internal fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256")
    .digest(bytes).joinToString("") { "%02x".format(it.toInt() and 255) }

/** Minimal strict JSON reader for embedded SYL2 metadata; no Python or JSON library dependency. */
private class MetadataJson(private val text: String) {
    private var at = 0
    private fun space() { while (at < text.length && text[at].isWhitespace()) at++ }
    private fun expect(c: Char) { space(); require(at < text.length && text[at++] == c) { "invalid JSON" } }
    private fun string(): String {
        expect('"')
        val out = StringBuilder()
        while (at < text.length) {
            val c = text[at++]
            if (c == '"') return out.toString()
            if (c == '\\') {
                require(at < text.length)
                when (val escaped = text[at++]) {
                    '"', '\\', '/' -> out.append(escaped)
                    'n' -> out.append('\n'); 'r' -> out.append('\r'); 't' -> out.append('\t')
                    'b' -> out.append('\b'); 'f' -> out.append('\u000c')
                    'u' -> { require(at + 4 <= text.length); out.append(text.substring(at, at + 4).toInt(16).toChar()); at += 4 }
                    else -> error("invalid JSON escape")
                }
            } else { require(c.code >= 32); out.append(c) }
        }
        error("unterminated JSON string")
    }
    private fun value(depth: Int = 0): Any? {
        require(depth <= 32) { "metadata nesting limit" }
        space(); require(at < text.length)
        return when (text[at]) {
            '"' -> string()
            '{' -> {
                at++; val map = linkedMapOf<String, Any?>(); space()
                if (text[at] != '}') while (true) {
                    val key = string(); require(!map.containsKey(key)); expect(':'); map[key] = value(depth + 1); space()
                    if (text[at] != ',') break
                    at++
                }
                expect('}'); map
            }
            '[' -> {
                at++; val list = arrayListOf<Any?>(); space()
                if (text[at] != ']') while (true) {
                    list.add(value(depth + 1)); space(); if (text[at] != ',') break; at++
                }
                expect(']'); list
            }
            else -> {
                val start = at
                while (at < text.length && text[at] !in ",]} \t\r\n") at++
                when (val token = text.substring(start, at)) {
                    "true" -> true; "false" -> false; "null" -> null
                    else -> token.toDouble().also { require(it.isFinite()) }
                }
            }
        }
    }
    @Suppress("UNCHECKED_CAST")
    fun parse(): Map<String, Any?> {
        val result = value() as Map<String, Any?>; space(); require(at == text.length); return result
    }
}

internal data class Tensor(val shape: IntArray, val data: FloatArray)

internal class SemanticWeights(file: File, task: String) {
    val hash: String
    val metadata: Map<String, Any?>
    val tensors: Map<String, Tensor>
    init {
        require(file.length() in 48..16_777_216) { "semantic artifact size limit" }
        val raw = file.readBytes()
        hash = sha256(raw)
        val body = raw.copyOfRange(0, raw.size - 32)
        require(MessageDigest.getInstance("SHA-256").digest(body).contentEquals(raw.copyOfRange(raw.size - 32, raw.size))) { "SYL2 checksum mismatch" }
        val b = ByteBuffer.wrap(body).order(ByteOrder.LITTLE_ENDIAN)
        fun bytes(n: Int): ByteArray { require(n >= 0 && n <= b.remaining()); return ByteArray(n).also { b.get(it) } }
        fun str(): String = String(bytes(b.int), Charsets.UTF_8)
        require(String(bytes(4), Charsets.US_ASCII) == "SYL2" && b.int == 1) { "invalid SYL2 header" }
        metadata = MetadataJson(str()).parse()
        require(metadata["schema"] == "semantic-records-v1" && metadata["taskId"] == task) { "incompatible semantic artifact" }
        require(metadata["featureNames"] == SemanticModelProvider.FEATURE_NAMES) { "feature order mismatch" }
        val result = linkedMapOf<String, Tensor>()
        val count = b.int; require(count in 1..100)
        repeat(count) {
            val name = str(); require(!result.containsKey(name))
            require(b.get().toInt() == 1) { "only float32 tensors supported" }
            val rank = b.get().toInt() and 255; require(rank in 1..4)
            val shape = IntArray(rank) { b.int.also { require(it in 1..65536) } }
            var elements = 1L
            for (dim in shape) { elements *= dim; require(elements <= 4_000_000) }
            require(b.long == elements * 4 && elements * 4 <= b.remaining())
            result[name] = Tensor(shape, FloatArray(elements.toInt()) { b.float.also { require(it.isFinite()) } })
        }
        require(!b.hasRemaining())
        tensors = result
    }
    fun tensor(name: String, vararg dimensions: Int): FloatArray {
        val tensor = tensors.getValue(name)
        require(tensor.shape.contentEquals(dimensions)) { "wrong tensor shape for $name" }
        return tensor.data
    }
    fun trained(): Boolean = (metadata["trainedSteps"] as Number).toInt() > 0
}

internal fun softmax(input: FloatArray): FloatArray {
    val max = input.maxOrNull() ?: return FloatArray(0)
    val out = FloatArray(input.size) { exp((input[it] - max).toDouble()).toFloat() }
    val sum = out.sum()
    return FloatArray(out.size) { out[it] / sum }
}
private fun argmax(a: FloatArray): Int = a.indices.maxByOrNull { a[it] } ?: 0
private fun linear(input: FloatArray, w: FloatArray, bias: FloatArray): FloatArray = FloatArray(bias.size) { row ->
    var sum = bias[row]; val offset = row * input.size
    for (i in input.indices) sum += w[offset + i] * input[i]
    sum
}

/** Mirrors PyTorch GRU r/z/n gate order, including reset-after hidden bias. */
private class GruDirection(weights: SemanticWeights, suffix: String, val inputSize: Int) {
    private val wi = weights.tensor("encoder.weight_ih_$suffix", 192, inputSize)
    private val wh = weights.tensor("encoder.weight_hh_$suffix", 192, 64)
    private val bi = weights.tensor("encoder.bias_ih_$suffix", 192)
    private val bh = weights.tensor("encoder.bias_hh_$suffix", 192)
    fun run(input: Array<FloatArray>, reverse: Boolean): Array<FloatArray> {
        var state = FloatArray(64)
        val output = Array(input.size) { FloatArray(64) }
        for (step in input.indices) {
            val index = if (reverse) input.lastIndex - step else step
            val x = linear(input[index], wi, bi)
            val h = linear(state, wh, bh)
            val next = FloatArray(64) { j ->
                val r = 1.0 / (1.0 + exp(-(x[j] + h[j]).toDouble()))
                val z = 1.0 / (1.0 + exp(-(x[64 + j] + h[64 + j]).toDouble()))
                val n = tanh(x[128 + j] + r * h[128 + j])
                ((1.0 - z) * n + z * state[j]).toFloat()
            }
            state = next; output[index] = next
        }
        return output
    }
}

class SemanticDetector(file: File, val role: String) {
    internal val weights = SemanticWeights(file, role)
    val sequenceBytes = (weights.metadata["sequenceBytes"] as Number).toInt()
    private val embedding = weights.tensor("embedding.weight", 259, 64)
    private val directions = (0..1).map { layer ->
        Pair(GruDirection(weights, "l$layer", if (layer == 0) 64 else 128),
             GruDirection(weights, "l${layer}_reverse", if (layer == 0) 64 else 128))
    }
    private val tw = weights.tensor("occurrence_head.weight", 3, 128)
    private val tb = weights.tensor("occurrence_head.bias", 3)
    private val kw = weights.tensor("kind_head.weight", 8, 128)
    private val kb = weights.tensor("kind_head.bias", 8)
    private val languageCount = (weights.metadata["languages"] as List<*>).size
    val languages: List<String> = (weights.metadata["languages"] as List<*>).map { it as String }
    private val lw = weights.tensor("language_head.weight", languageCount, 128)
    private val lb = weights.tensor("language_head.bias", languageCount)
    init { require(role in listOf("definitions", "usages") && sequenceBytes >= 4) }
    internal fun forward(ids: IntArray): Triple<Array<FloatArray>, Array<FloatArray>, FloatArray> {
        var output = Array(ids.size) { embedding.copyOfRange(ids[it] * 64, ids[it] * 64 + 64) }
        for ((forward, backward) in directions) {
            val f = forward.run(output, false); val b = backward.run(output, true)
            output = Array(ids.size) { f[it] + b[it] }
        }
        val mean = FloatArray(128) { j -> output.sumOf { it[j].toDouble() }.toFloat() / ids.size }
        return Triple(Array(ids.size) { linear(output[it], tw, tb) },
                      Array(ids.size) { linear(output[it], kw, kb) }, linear(mean, lw, lb))
    }
    fun detect(text: String, documentId: String, revision: String): Pair<List<SymbolResult>, FloatArray> {
        require(documentId.isNotEmpty() && revision.isNotEmpty())
        val units = arrayListOf(0); val bytes = arrayListOf(0)
        var unit = 0; var byte = 0
        while (unit < text.length) {
            val cp = text.codePointAt(unit)
            require(cp !in 0xD800..0xDFFF) { "unpaired surrogate in source" }
            unit += Character.charCount(cp)
            byte += when { cp < 128 -> 1; cp < 2048 -> 2; cp < 65536 -> 3; else -> 4 }
            units.add(unit); bytes.add(byte)
        }
        val raw = text.toByteArray(Charsets.UTF_8)
        val chars = units.size - 1
        val tags = IntArray(chars); val confidence = FloatArray(chars)
        val kinds = Array(chars) { FloatArray(8) }; val language = FloatArray(languageCount)
        var a = 0
        while (a < chars) {
            var b = a + 1
            while (b < chars && bytes[b + 1] - bytes[a] <= sequenceBytes) b++
            val input = IntArray(bytes[b] - bytes[a] + 1) { if (it == 0) 256 else raw[bytes[a] + it - 1].toInt() and 255 }
            val (tagLogits, kindLogits, langLogits) = forward(input)
            val lang = softmax(langLogits)
            for (j in lang.indices) language[j] += lang[j] * (b - a)
            for (i in a until b) {
                val index = bytes[i] - bytes[a] + 1
                val probs = softmax(tagLogits[index]); tags[i] = argmax(probs)
                confidence[i] = probs[tags[i]]; kinds[i] = kindLogits[index]
            }
            a = b
        }
        if (chars > 0) for (j in language.indices) language[j] /= chars
        val records = arrayListOf<SymbolResult>(); var start = -1
        for (i in 0..chars) {
            val tag = if (i == chars) 0 else tags[i]
            if (start >= 0 && tag != 2) {
                val k = FloatArray(8); var certainty = 0f
                for (j in start until i) { for (c in k.indices) k[c] += kinds[j][c]; certainty += confidence[j] }
                val first = units[start]; val end = units[i]
                records.add(SymbolResult("$role:$first:$end", documentId, revision, first, end,
                                         text.substring(first, end), argmax(k), certainty / (i - start), text.length))
                start = -1
            }
            if (i < chars && tag != 0 && start < 0) start = i
        }
        return Pair(records, language)
    }
}

class SemanticModelProvider(definitionsFile: File, usagesFile: File, intelligenceFile: File) {
    private val definitions = SemanticDetector(definitionsFile, "definitions")
    private val usages = SemanticDetector(usagesFile, "usages")
    private val intelligence = SemanticWeights(intelligenceFile, "code-intelligence")
    private val hw = intelligence.tensor("hidden.weight", 64, FEATURE_NAMES.size)
    private val hb = intelligence.tensor("hidden.bias", 64)
    private val sw = intelligence.tensor("score.weight", 1, 64)
    private val sb = intelligence.tensor("score.bias", 1)
    val languages: List<String> get() = definitions.languages
    init {
        val dependencies = intelligence.metadata["dependencies"] as Map<*, *>
        require(dependencies["definitions"] == definitions.weights.hash && dependencies["usages"] == usages.weights.hash) {
            "intelligence was trained with different detector artifacts"
        }
        require(definitions.weights.metadata["registryHash"] == usages.weights.metadata["registryHash"] &&
                definitions.weights.metadata["registryHash"] == intelligence.metadata["registryHash"])
        require(definitions.languages == usages.languages && definitions.languages == intelligence.metadata["languages"])
    }
    fun analyze(text: String, documentId: String, cursor: Int? = null): SemanticSnapshot {
        require(definitions.weights.trained() && usages.weights.trained()) { "detectors have no training steps" }
        if (cursor != null) {
            require(cursor in 0..text.length)
            require(cursor == 0 || cursor == text.length || !(text[cursor - 1].isHighSurrogate() && text[cursor].isLowSurrogate()))
        }
        val context = if (cursor == null) text else text.substring(0, cursor)
        val revision = sha256(text.toByteArray(Charsets.UTF_8))
        val (defs, dl) = definitions.detect(context, documentId, revision)
        val (uses, ul) = usages.detect(context, documentId, revision)
        return SemanticSnapshot(documentId, revision, defs, uses, dl, ul)
    }
    /** Cursor is in usage.documentId. A completion query accepts only prefix-derived local records. */
    fun resolve(usage: SymbolResult, candidates: List<SymbolResult>, cursor: Int? = usage.start,
                threshold: Float = .8f, completionContext: Boolean = false): DefinitionResolution {
        require(intelligence.trained()) { "intelligence has no trained link examples" }
        require(threshold in 0f..1f && candidates.size <= 512)
        require(candidates.map { Triple(it.documentId, it.revision, it.id) }.toSet().size == candidates.size)
        if (completionContext) {
            require(cursor != null && usage.end <= cursor) { "usage is after completion cursor" }
            require(usage.contextEnd != null && usage.contextEnd <= cursor) { "usage was not detected from prefix-only context" }
            require(candidates.none { it.documentId == usage.documentId &&
                (it.end > cursor || it.contextEnd == null || it.contextEnd > cursor) }) { "candidate was not detected from prefix-only context" }
        }
        val scores = FloatArray(candidates.size + 1) { index ->
            val features = features(usage, candidates.getOrNull(index), cursor)
            val hidden = linear(features, hw, hb).map { tanh(it) }.toFloatArray()
            linear(hidden, sw, sb)[0]
        }
        val probs = softmax(scores); val best = argmax(probs)
        val target = if (best < candidates.size && probs[best] >= threshold) candidates[best] else null
        return DefinitionResolution(usage, target, probs[best], probs)
    }
    companion object {
        val KINDS = listOf("unknown", "variable", "function", "type", "field", "parameter", "module", "constant")
        val FEATURE_NAMES = listOf("null", "sameDocument", "sameName", "sameKind", "usageConfidence",
            "definitionConfidence", "definitionBeforeUsage", "signedDistance") +
            listOf("usage", "definition").flatMap { role -> KINDS.map { "$role" + "Kind:$it" } } +
            listOf("queryPresent", "usagePosition", "definitionPosition", "queryPosition", "usageToQuery",
                   "absoluteUsageToQuery", "definitionToQuery", "absoluteDefinitionToQuery")
        fun features(u: SymbolResult, d: SymbolResult?, cursor: Int? = null): FloatArray {
            require(cursor == null || cursor >= 0)
            val same = d != null && u.documentId == d.documentId
            require(!same || u.revision == d!!.revision) { "stale definition revision" }
            fun flag(b: Boolean) = if (b) 1f else 0f
            fun bounded(p: Int) = p.toDouble().div(p.toDouble() + 4096.0).toFloat()
            fun distance(p: Int) = if (cursor != null) ((p.toDouble() - cursor) / 4096.0).coerceIn(-1.0, 1.0).toFloat() else 0f
            val uf = distance(u.start); val df = if (same) distance(d!!.start) else 0f
            return (listOf(flag(d == null), flag(same), flag(d != null && u.name == d.name),
                flag(d != null && u.kind == d.kind), u.confidence, d?.confidence ?: 0f,
                flag(same && d!!.start <= u.start), if (same) ((d!!.start.toDouble() - u.start) / 4096.0).coerceIn(-1.0, 1.0).toFloat() else 0f) +
                (0..7).map { flag(u.kind == it) } + (0..7).map { flag(d != null && d.kind == it) } +
                listOf(flag(cursor != null), bounded(u.start), if (same) bounded(d!!.start) else 0f,
                       cursor?.let { bounded(it) } ?: 0f, uf, abs(uf), df, abs(df))).toFloatArray()
        }
    }
}
