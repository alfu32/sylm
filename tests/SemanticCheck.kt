package syntaxlm

import java.io.File
import kotlin.system.measureNanoTime

/** Cross-runtime parity probe and repeatable warm Kotlin latency benchmark. */
object SemanticCheck {
    @JvmStatic fun main(args: Array<String>) {
        val root = File(args[0])
        val definitions = SemanticDetector(File(root, "definitions.model.bin"), "definitions")
        val usages = SemanticDetector(File(root, "usages.model.bin"), "usages")
        val provider = SemanticModelProvider(File(root, "definitions.model.bin"), File(root, "usages.model.bin"), File(root, "code-intelligence.model.bin"))
        if (args.size > 1) {
            val source = File(args[1]).readText()
            val samples = 30
            for ((name, detector) in listOf("definitions" to definitions, "usages" to usages)) {
                repeat(5) { detector.detect(source, "benchmark", "revision") }
                val timings = List(samples) { measureNanoTime { detector.detect(source, "benchmark", "revision") } / 1e6 }.sorted()
                println("{\"task\":\"$name\",\"utf16Units\":${source.length},\"medianMs\":${timings[samples / 2]},\"p95Ms\":${timings[28]}}")
            }
            val u = SymbolResult("use", "bench", "rev", 2000, 2004, "name", 1, .9f, 2010)
            val candidates = (0 until 100).map { SymbolResult("d$it", "bench", "rev", it * 10, it * 10 + 4, "name", 1, .9f, 2010) }
            repeat(10) { provider.resolve(u, candidates, 2010) }
            val times = List(samples) { measureNanoTime { provider.resolve(u, candidates, 2010) } / 1e6 }.sorted()
            println("{\"task\":\"intelligence\",\"candidates\":100,\"medianMs\":${times[samples / 2]},\"p95Ms\":${times[28]}}")
            return
        }
        val (tags, kinds, language) = definitions.forward(intArrayOf(256, 97, 0, 255, 195, 169))
        fun floats(v: FloatArray) = v.joinToString(",", "[", "]")
        println(tags.joinToString(",", "[", "]") { floats(it) })
        println(kinds.joinToString(",", "[", "]") { floats(it) })
        println(floats(language))
        val text = "# 😀\nαx = 1\nuse(αx)\n"
        val (records, _) = definitions.detect(text, "doc", "rev")
        println(records.joinToString(",", "[", "]") { "[${it.start},${it.end},${it.kind},${it.confidence},${it.contextEnd}]" })
        val usage = SymbolResult("u", "doc", "rev", 88, 91, "box", 1, .7f, 100)
        val candidate = SymbolResult("d", "doc", "rev", 4, 7, "box", 1, .9f, 100)
        println(floats(SemanticModelProvider.features(usage, candidate, 100)))
        println(floats(provider.resolve(usage, listOf(candidate), 100, 0f, true).candidateProbabilities))
        check(runCatching { provider.resolve(usage, listOf(candidate.copy(revision = "old"))) }.isFailure)
        check(runCatching { provider.resolve(usage.copy(contextEnd = 120), listOf(candidate), 100, completionContext = true) }.isFailure)
        check(provider.resolve(usage, emptyList(), 100).definition == null)
        val prefix = provider.analyze(text, "doc", 5)
        check((prefix.definitions + prefix.usages).all { it.end <= 5 && it.contextEnd == 5 })
        check(runCatching { provider.analyze(text, "doc", 3) }.isFailure) // inside emoji surrogate pair
    }
}
