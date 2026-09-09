# Three-model architecture

Status: revised implementation specification, incorporating the requirement that local models do the analysis and the editor gathers results. The repository currently contains only the legacy perceptron highlighter and its SYLM v1 Kotlin runner. The three providers and neural trainers below are proposed work, not existing functionality.

## Review decisions

The three-task split is sound, but the earlier proposal left several requirements unresolved:

| Issue | Decision |
| --- | --- |
| A bounded n-gram ignores most of the prefix | Model 3 starts with a causal GRU that processes the whole prefix; n-grams are evaluation baselines only. |
| “GRU or Transformer” leaves runtime math unspecified | Use GRUs for the first neural implementation, with explicit dimensions and gate conventions below. |
| One flat label sequence cannot represent nested arrays and objects | Model 1 has separate lexical and nested-region outputs. |
| Definitions need to be displayed as well as selected | Store name, signature, and full declaration ranges, plus the document revision. |
| A role classifier alone cannot resolve references | Model 2 learns declaration/scope spans and links usages to dynamically predicted definitions. |
| Language-specific client logic duplicates the models' work | Use lossless byte inputs and learned boundaries/roles/links, with generic decoding inside the library. |
| An append-only completion API misses ordinary editor changes | Use immutable document snapshots, cursor positions, invalidation, and replay. |
| A single matrix omits vocabulary, biases, and recurrent operations | Export a self-contained binary tensor container for each task. |

Python training may use PyTorch. Kotlin/JVM inference uses the exported artifacts, Kotlin/JVM libraries, and local inference operators. Python is not installed or invoked by the client. Maintaining a dependency-free *trainer* is not a requirement and should not force weaker models.

## Application boundary: submit source, receive results

The editor supplies an immutable source snapshot, document identity/revision, language, and cursor or snippet range. It receives lexical spans, nested regions, definitions, usages with target links, and completion candidates. It does not tokenize, classify, parse scopes, rank reference targets, or maintain an analysis symbol table.

The bundled Kotlin inference library performs tensor operations and generic decoding. Learned predictions drive language-sensitive analysis. The library also validates ranges, maps positions, caches document results, and rejects stale results.

~~~text
Python: corpus → train three neural models → export three self-contained binaries
Editor: snapshot → LocalCodeModels Kotlin library → ranges, links, completions
                              |
                   model weights + generic inference
~~~

A network does not itself return editor objects: a decoder converts scores into ranges, links, and strings. That decoder belongs inside the library. There is no required handwritten language grammar, keyword classifier, or deterministic scope resolver in the first model-driven implementation.

Definition links are predictions. Model 2 can abstain or report ambiguity, but cannot promise compiler-equivalent navigation for every language. An optional compiler verification backend can be added later without changing the editor contract.

## Shared source and runtime contract

### Original document and snippets

All public ranges and new training annotations use UTF-16 offsets, inclusive start and exclusive end. Offsets index the unchanged source: no newline normalization, whitespace stripping, or Unicode normalization. Line and column are zero-based; columns use UTF-16 units. An LSP adapter must honor the position encoding negotiated with its client.

~~~kotlin
data class DocumentId(val uri: String, val revision: Long)
data class SourceRange(val start: Int, val end: Int)
data class DocumentSnapshot(
    val id: DocumentId,
    val language: String,
    val text: String,
)
data class SourceLocation(val document: DocumentId, val range: SourceRange)
~~~

Build Python codepoint-to-UTF-16 and UTF-8-byte-to-UTF-16 maps once per snapshot. Never locate an occurrence by searching for its spelling. Validate bounds and scalar boundaries. A range must not split a surrogate pair. For a temporarily unpaired surrogate in an editor buffer, use a documented replacement-byte view for inference while retaining its original UTF-16 position.

A snippet carries its original DocumentId and baseOffsetUtf16. Convert local [s,e) to original [base+s,base+e). Verify that the snippet is the corresponding substring of that revision. A detached snippet can give valid locations but has limited surrounding context; report that limitation in the result. Prefer analyzing a document snapshot and querying spans intersecting the snippet. Do not clip a complete comment or array into a falsely complete region.

Results carry the input revision. After an edit, stale results must be discarded or explicitly remapped by the editor. Source text is obtained from the matching snapshot; repeated copies of large nested substrings are optional convenience values, not stored in every span.

### Shared lossless input

All three models consume UTF-8 bytes of the original source plus BOS and a language ID. Preserve whitespace, line breaks, comments, and literal contents. No language lexer, keyword lists, Python tokenize call, or normalization participates in neural inference.

A portable, language-independent encoder maps bytes back to UTF-16 ranges. Structural heads operate only at Unicode scalar boundaries; continuation-byte positions are masked during span decoding. Invalid intermediate editor text follows the replacement-byte rule above.

The input vocabulary consists of 256 bytes plus BOS, EOS, PAD, and supported language IDs. Unknown languages use an explicit unknown-language ID and report unsupported/unvalidated coverage. Exact vocabulary IDs and their hashes are exported with each artifact and tested across Python and Kotlin.

### Common neural encoder for models 1 and 2

Share implementation and architecture, initially with independent weights per artifact:

| Layer | Proposed shape | Purpose |
| --- | --- | --- |
| Byte/special-token embedding | [V,64] | Encodes lossless source, including arbitrary identifier spellings. |
| Context GRU layer 1 | bidirectional, input 64, H=64 per direction | Learns spelling, delimiters, lexical modes, and syntax from both directions. |
| Context GRU layer 2 | bidirectional, input 128, H=64 per direction | Produces a 128-vector at each source byte using wider context. |
| Task heads | described below | Predict lexical spans, nested regions, declarations, and links. |

For each decoded interval, its representation is the mean of the 128-vectors inside it. Prefix sums make pooling efficient. Source coordinates remain outside the learned network and are never approximated by it.

Process the whole analysis unit initially. Long documents can be processed in chunks while carrying forward/backward states across chunk boundaries; reset at document boundaries, not arbitrary chunk boundaries. A bounded-context optimization must expose its coverage and be evaluated separately.

These dimensions are initial engineering choices, not measured quality or latency claims. Byte input trades a simple portable tokenizer for longer sequences. Source length affects work and activation memory. Weights are immutable and reusable; request/session state is private to each document analysis.

## Model 1: lexical roles and nested source regions

### Outputs and label semantics

The requested categories are retained:

~~~text
KEYWORD, IDENTIFIER, PUNCTUATION, STRING_LITERAL, NUMBER_LITERAL,
ARRAY_LITERAL, OBJECT_LITERAL, SINGLE_LINE_COMMENT, MULTILINE_COMMENT
~~~

Add OPERATOR, BOOLEAN_LITERAL, NULL_LITERAL, and UNKNOWN to avoid forcing ordinary source into a wrong requested category. The annotation specification defines literal semantics for each language; the model learns them from labeled examples. A Python list may map to ARRAY_LITERAL and a dictionary to OBJECT_LITERAL; sets, blocks, indexing, and constructor calls must not be labeled as those literals just because they contain brackets or construct a collection.

There are two outputs:

- Lexical spans: ordered, non-overlapping outer lexemes with exact boundaries. Adjacent equal labels are not automatically merged.
- Region spans: nested structural ranges with parent IDs, including arrays, objects, and interpolated expressions/literals where necessary.

Interpolated strings keep an outer STRING_LITERAL range and nested expression spans, preserving identifiers used inside them.

A complete comment covers its delimiters and content. A single-line comment excludes its terminating newline; a block comment uses MULTILINE_COMMENT even if its text happens to fit on one line. An unterminated comment/string extends to the available input boundary with complete=false.

### Layers and decoding

1. **Lossless input and embedding:** encode bytes with the common embedding table. The network sees comment delimiters, spaces, quote characters, and names directly.
2. **Bidirectional context encoder:** produces a 128-vector per byte and learns lexical modes/contextual keywords using both sides of a position.
3. **Lexical segmentation head:** linear [4×Clex+1,128] plus bias predicts BIOES role tags at Unicode scalar positions: beginning, inside, end, single, plus outside/trivia. Clex is the exported lexical-role count (11 for the proposed roles, excluding arrays/objects). A constrained sequence decoder enforces valid transitions and combines complete multiword/multiline lexemes. A separate [1,128] completeness head scores pooled spans; an open sequence at the document end may decode as incomplete rather than inventing a closing delimiter.
4. **Region boundary heads:** independent projections score starts, ends, type, and completeness for ARRAY_LITERAL, OBJECT_LITERAL, STRING_LITERAL, and INTERPOLATION_EXPRESSION. Keep multiple candidate boundaries to support nesting.
5. **Region pairing head:** concatenate opening, closing, and mean interior representations (3×128) plus 16 generic length/boundary features. Apply linear [64,400], tanh, linear [5,64] plus biases. Outcomes are the four region kinds plus NONE. Missing ends use a virtual document-end candidate with a completeness flag. Learn pairing rather than infer it from bracket counting.
6. **Generic span decoder:** select compatible scored regions, reject crossing ranges, and derive parent IDs by containment. Candidate pruning and beam limits are artifact configuration; report incomplete/pruned coverage. The head cannot recover a region whose boundary was discarded.

Language meaning comes from annotations and learned scores. Decoding enforces coordinate and nesting invariants only. A flat BIOES output covers outer lexemes; region heads expose interpolation expressions, and model 2 independently detects identifier occurrences within them.

~~~kotlin
data class LexicalSpan(
    val id: Int,
    val location: SourceLocation,
    val role: String, // validated against artifact labels
    val complete: Boolean,
)
data class RegionSpan(
    val id: Int,
    val parentId: Int?,
    val location: SourceLocation,
    val role: String,
    val complete: Boolean,
)
data class SourceRoleResult(
    val document: DocumentId,
    val lexical: List<LexicalSpan>,
    val regions: List<RegionSpan>,
    val contextComplete: Boolean,
    val candidateSearchLimited: Boolean,
)
~~~

Parent IDs are result-local. Display styling prefers a lexical child over its enclosing aggregate region; highlighting an entire array must not obscure its identifiers or comments.

### Training and evaluation

Use lexer/parser-derived annotations offline plus explicit incomplete-source examples. These tools are training dependencies only. Keep lexical labels and regions separate. Loss combines lexical tag cross-entropy, boundary/completeness loss, and region pair classification, masking unannotated positions. Include negative boundary pairs and block/indexing examples. The example below is partial; omitted labels are not silently assigned a heuristic ground truth.

~~~json
{
  "language": "javascript",
  "offsetEncoding": "utf16",
  "annotationCoverage": "partial",
  "code": "const xs = [1, 2]; /* values */",
  "lexical": [
    {"start": 0, "end": 5, "role": "KEYWORD"},
    {"start": 19, "end": 31, "role": "MULTILINE_COMMENT"}
  ],
  "regions": [
    {"id": "r0", "parentId": null, "start": 11, "end": 17, "role": "ARRAY_LITERAL"}
  ]
}
~~~

Measure exact lexical span F1, nested region F1, candidate recall, and performance on incomplete code. Compare against a lexer/parser baseline and the existing perceptron. Testing only on the heuristic labels used to bootstrap training does not establish learned contextual ability.

## Model 2: identifier definitions, usages, and links

### Inputs and semantic scope

Model 2 independently encodes source bytes. Model-1 candidate spans/probabilities are optional observations; its own identifier-boundary head can recover a candidate missed by model 1. A wrong model-1 label must not permanently eliminate a potential identifier.

A symbol kind (function, variable, parameter, property, type) differs from an occurrence role. A property can be defined or used. Imports can define an alias and reference another symbol. Model these facts on separate axes.

Declaration, scope, visibility, aliasing, and usage relationships are learned from semantic annotations. The Kotlin client has no deterministic scope resolver. Single-document links are the first delivery slice. Cross-file links require other documents or a versioned library-owned index as input; weights cannot know definitions in files never supplied.

### Layers

1. **Contextual encoder:** independent instance of the common 128-output byte encoder.
2. **Identifier-boundary head:** linear [5,128] predicts BIOES/OUTSIDE identifier spans, including occurrences inside interpolation expressions.
3. **Occurrence heads:** mean-pool each candidate and apply linear [4,128] for DEFINITION, USAGE, BOTH, UNKNOWN. Separate linear [K,128] predicts symbol kind. Import/alias, read/write, and member access are separate learned flags.
4. **Declaration and scope pointer heads:** project occurrence/start/end byte vectors to 64 dimensions and score endpoint candidates using dot products. Learn name-to-signature/declaration ranges, enclosing scopes, and optional parent-scope links. Generic decoding validates containment; incomplete predictions retain null/provisional fields.
5. **Definition candidates:** collect predicted definitions plus a null candidate. For small documents score every usage/definition pair. For large documents use learned query/key similarity to shortlist candidates. Do not impose nearest-spelling/visibility heuristics that discard aliases or forward declarations. Measure retrieval recall and report limits.
6. **Learned link head:** concatenate usage vector 128, definition vector 128, elementwise product 128, and 16 generic distance/span/learned-scope features. Score using linear [64,400], tanh, linear [1,64]. A learned null score competes with definitions. Candidate IDs are dynamic pointers into this analysis, not fixed symbol classes learned during training.
7. **Result decoder and lookup index:** calibrated scores/margins determine RESOLVED, AMBIGUOUS, UNRESOLVED, or EXTERNAL. Preserve alternatives and provenance. A high softmax score alone does not verify a link; model-only targets are marked predicted.

Spelling and scope can contribute learned features. No handwritten language-specific resolver is required. Hard negatives include same-name symbols in different scopes. Retrieval, thresholds, and learned scope quality directly limit navigation accuracy.

### Editor contract

~~~kotlin
data class DefinitionLocation(
    val symbolId: String, // unique within the workspace analysis generation
    val document: DocumentId,
    val name: String,
    val kind: String,
    val nameRange: SourceRange,
    val signatureRange: SourceRange?,
    val declarationRange: SourceRange?, // null if the full declaration is unknown
)
data class IdentifierOccurrence(
    val location: SourceLocation,
    val name: String,
    val role: String,
    val symbolKind: String,
)
data class DefinitionCandidate(
    val definition: DefinitionLocation,
    val score: Float,
    val verified: Boolean, // false for model-only links; optional external verification
)
data class DefinitionQueryResult(
    val origin: SourceLocation,
    val status: String,
    val candidates: List<DefinitionCandidate>,
)
~~~

Navigate/select using nameRange. Present a definition using signatureRange or declarationRange from the matching snapshot. When present, the full declaration must contain nameRange. If unknown, show the available name/signature and mark the preview incomplete. This mirrors the distinction between targetRange and targetSelectionRange in the [LSP LocationLink specification](https://raw.githubusercontent.com/microsoft/language-server-protocol/gh-pages/_specifications/lsp/3.17/types/locationLink.md).

The provider exposes definitionAt(document, offset) and referencesOf(symbolId). Querying a definition itself should resolve to itself. References are complete only within the indexed workspace scope; report coverage. If another file has changed, invalidate its old symbols and dependent links. Document-local IDs are not permanent identities across edits.

### Training and evaluation

Obtain identifier, declaration, scope, occurrence, and reference-link annotations offline from compiler/semantic tooling, supplemented by checked annotations. A syntax tree alone cannot label all reference targets. Train boundary/role heads with classification losses, endpoint heads with pointer losses, and the link head with candidate cross-entropy including null/multiple-valid-target cases. Mask unavailable labels. Include same-name inner/outer definitions as hard negatives. Train/evaluate with predicted as well as gold candidates to expose cascading errors.

~~~json
{
  "documentUri": "file:///workspace/main.kt",
  "offsetEncoding": "utf16",
  "annotationCoverage": "partial",
  "code": "fun inc(count: Int) = count + 1",
  "definitions": [
    {"id": "d0", "nameStart": 4, "nameEnd": 7, "declarationStart": 0, "declarationEnd": 31, "name": "inc", "kind": "function"},
    {"id": "d1", "nameStart": 8, "nameEnd": 13, "declarationStart": 8, "declarationEnd": 18, "name": "count", "kind": "parameter"}
  ],
  "usages": [
    {"start": 22, "end": 27, "name": "count", "definitionId": "d1"}
  ]
}
~~~

Measure occurrence accuracy, candidate retrieval recall, exact target-link accuracy, incorrect-navigation rate, and resolved coverage. An always-unresolved model must not appear accurate. Test shadowing, forward references, imports, members, overloads, incomplete declarations, and external symbols; identify unsupported cases explicitly.

## Model 3: causal full-prefix completion

### Selected model and meaning of context

Use a two-layer unidirectional GRU. Feed every byte of the source before the cursor in order, carrying hidden state throughout the prefix. There is no default left truncation or fixed n-gram window.

The hidden state is a finite learned summary: all earlier bytes can influence a prediction, but exact recall of arbitrarily distant code is not guaranteed. A GRU is suitable for streaming the whole prefix with bounded recurrent state. A conventional finite-window Transformer would change the context contract and is not the initial implementation.

### Layers

| Layer | Proposed shape | Purpose |
| --- | --- | --- |
| Lossless byte tokenizer | 256 UTF-8 byte IDs plus BOS, EOS, PAD and language IDs | Retains spaces/newlines and represents unseen identifiers. Special IDs are fixed by artifact metadata. |
| Embedding | [V,96] | Maps each input byte/special ID to a learned vector. |
| Causal GRU layer 1 | input 96, hidden 192 | Updates prefix state from each embedding. |
| Causal GRU layer 2 | input 192, hidden 192 | Builds a second recurrent representation. |
| Vocabulary projection | [257,192] plus bias [257] | Predicts a next byte or EOS; BOS/PAD/language IDs are never generated. |
| Word decoder | beam search plus learned word-boundary head | Accumulates generated bytes into a word/token or identifier suffix. |

Feed BOS and the language ID before source bytes. Reset state between documents. GRU order is recurrent, so no positional embedding or attention KV cache is used. The two hidden vectors require 384 float32 values (1,536 bytes); weights, document text, editor checkpoints, and beam states require additional memory.

The initial unquantized model has 439,170 + 96×V parameters including embeddings, recurrent weights/biases, output projection, and a [1,192] word-boundary head with bias. This is a sizing calculation, not a performance result. Measure cold prefix-processing latency and warm next-word latency separately.

### Recurrent operator contract

Use the PyTorch GRU gate convention with row-major matrices, gates ordered reset/update/new, and separate input and recurrent biases. For one layer:

~~~text
r = sigmoid(W_ir x + b_ir + W_hr h + b_hr)
z = sigmoid(W_iz x + b_iz + W_hz h + b_hz)
n = tanh(W_in x + b_in + r * (W_hn h + b_hn))
h_new = (1 - z) * n + z * h
~~~

Here * is elementwise multiplication. Input weights have shape [3H,I], recurrent weights [3H,H], and each bias [3H]. The reset gate is applied after the recurrent affine transform of the candidate. Python export and Kotlin inference must use this exact variant. It differs from some other GRU conventions. See the [official GRU equations and tensor layouts](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRU.html).

The same operator serves the bidirectional byte encoders in models 1 and 2. Inference disables dropout and starts with zeros unless resuming a validated checkpoint.

### Next byte versus next word

The neural objective is next-byte prediction. The user-facing provider returns completed words/tokens:

- After each generated byte, a sigmoid word-boundary head predicts whether a complete word/token ended. Train these boundaries from offline lexical annotations. Stop at a sufficiently scored boundary, EOS, or a byte limit; branch in beam search when continuation is plausible.
- Return an identifier suffix when the cursor follows a partial identifier. Default replaceRange is [cursor,cursor); causal state and the boundary head determine continuation.
- Preserve proposed whitespace in insertText and never return invalid UTF-8.
- Include punctuation as a source token; do not equate one subtoken with one word.
- Clone hidden states for candidate beams. Predictions must not advance the committed document state.
- Rank by summed log probabilities, with any length normalization explicitly configured. Such scores are ranking scores, not calibrated probabilities of whole-word correctness.
- Flag a candidate cut short by a generation limit as incomplete.

### Editor API and state invalidation

~~~kotlin
data class CompletionCandidate(
    val insertText: String,
    val replaceRange: SourceRange,
    val logScore: Float,
    val complete: Boolean,
)
data class CompletionResult(
    val document: DocumentId,
    val cursorUtf16: Int,
    val processedPrefixUtf16: Int,
    val contextComplete: Boolean,
    val candidates: List<CompletionCandidate>,
)
interface CodeCompletionProvider {
    fun complete(
        snapshot: DocumentSnapshot,
        cursorUtf16: Int,
        topK: Int = 8,
    ): CompletionResult
    fun invalidate(documentUri: String)
}
~~~

Initial correctness can recompute the prefix. Optimization stores hidden-state checkpoints keyed by document revision, language, model hash, and UTF-16 position. After insertion/deletion/replacement, find a verified unchanged prefix, resume before the edit, and replay bytes through the cursor. Cursor movement backward must not expose future state. Bound checkpoint memory independently of document size.

If a large-prefix computation is cancelled or unfinished, return pending/incomplete context rather than claim the whole prefix was used. Cold full-prefix processing is linear in prefix length; there is no constant-latency guarantee.

### Training and evaluation

Train with next-byte cross-entropy plus word-boundary binary cross-entropy and truncated backpropagation through time. Carry the numerical hidden state between consecutive chunks from the same file while detaching gradients at chunk boundaries. Reset only at file boundaries; shuffled unrelated chunks cannot share state. Gradient truncation is a training limitation, distinct from discarding prefix input at inference.

Use project-separated, deduplicated code corpora with recorded provenance. Report held-out bits per byte and exact next-word top-k after decoding. Test long-distance examples with identical recent suffixes but different early declarations, and compare to an n-gram baseline. Compare fresh full-prefix inference against append and mid-document replay. Processing the entire prefix is a structural guarantee; successfully using distant information is an empirical property.

## Kotlin facade: application gathers results

The facade owns model loading, inference, result indexes, and per-document caches. A minimal API is:

~~~kotlin
data class SymbolAnalysis(
    val document: DocumentId,
    val definitions: List<DefinitionLocation>,
    val occurrences: List<IdentifierOccurrence>,
    val links: List<DefinitionQueryResult>,
    val contextComplete: Boolean,
)
data class CodeAnalysis(
    val document: DocumentId,
    val roles: SourceRoleResult,
    val symbols: SymbolAnalysis,
)
interface LocalCodeModels {
    fun analyze(snapshot: DocumentSnapshot): CodeAnalysis
    fun definitionAt(snapshot: DocumentSnapshot, offsetUtf16: Int): DefinitionQueryResult?
    fun complete(snapshot: DocumentSnapshot, cursorUtf16: Int, topK: Int = 8): CompletionResult
    fun invalidate(documentUri: String)
}
~~~

The editor loads the three binaries once, submits snapshots on a background worker, and applies results only when revisions match. Highlighting uses roles; clicking an identifier uses definitionAt; a definition preview slices the matching snapshot using declarationRange; typing uses complete. The app supplies snapshots for any external documents it wants analyzed, but the library owns their analysis index.

The facade may lazily run only the necessary models and reuse cached results. Model 3 does not wait for models 1 or 2. Cancelling a request leaves committed document state valid. This initial synchronous interface makes threading the host's choice; a suspending adapter can wrap it without changing inference semantics.

## Binary export and Kotlin independence

Each neural artifact is a single binary file containing all learned weights and preprocessing tables it needs:

~~~text
token-role.model.bin
identifier-relation.model.bin
next-word.model.bin
~~~

Use a new SYL2 container; leave legacy SYLM v1 readable only by its legacy loader. Proposed physical layout, all integers little-endian:

~~~text
magic[4] = SYL2
containerVersion:u32 = 1
metadataLength:u32
metadata[metadataLength] = UTF-8 JSON
tensorCount:u32
repeat tensorCount:
    nameLength:u32, nameUtf8[nameLength]
    dtype:u8 = 1 (float32)
    rank:u8
    dimensions[rank]:u32
    payloadBytes:u64
    payload[payloadBytes] = contiguous row-major float32
checksum[32] = SHA-256 of preceding bytes
~~~

Metadata includes task ID, architecture/operator version, tensor names/shapes, language support, byte-encoding and output-decoding versions, vocabulary/labels, output head schemas, normalization rules, thresholds, and training provenance. Each artifact includes its required encoding/decoding configuration; Kotlin needs no adjacent Python files or training checkpoint.

The loader validates task, architecture, dimensions, byte counts, duplicate names, finite values, checksum, and allocation limits before using tensors. Explicitly define metadata fields and fixtures before implementing loaders. Do not load Python pickle/Torch checkpoints in Kotlin. Session states are runtime values, not model weights exported from training.

Python's reference inference must reload exported float32 artifacts for comparison; comparing Kotlin against unexported float64 training weights hides export drift. Pin a tested PyTorch training dependency when implementing the trainer; the current standard-library-only requirements file describes only the legacy CLI.

## Implementation order and acceptance gates

1. Implement shared snapshot/range types, byte-encoding specification, SYL2 schema, and Python/Kotlin fixtures. Start neural work only after matching byte IDs and coordinates.
2. Implement the Python trainer/exporter and Kotlin GRU/linear operators. Compare single-step and sequence states, reverse-direction outputs, and exported logits.
3. Deliver model 1 lexical and nested-region training/inference, with exact offset tests and parser/scanner baselines.
4. Deliver model 2 identifier, declaration/scope, and link training/inference. Declare validated semantic coverage per language and publish incorrect-link rates.
5. Deliver the causal GRU completion trainer and Kotlin word decoder with full-prefix replay; an n-gram benchmark does not fulfill this delivery.
6. Package three independent providers, runnable examples, model metadata, and held-out evaluation reports.

Cross-runtime tests must cover comments, triple/raw/interpolated strings, operators, Unicode including emoji before a range, CRLF, nested arrays/objects, incomplete input, repeated spellings, and edits. Require exact tokenizer IDs/ranges and decoded structure, with documented float32 numerical tolerances for logits. Near ties must be identified instead of asserting bit-identical probabilities.

Model artifacts and nine bootstrap snippets alone do not establish editor quality. Report actual held-out results, model sizes, device/JVM, peak memory, and cold/warm latency before making quality/performance claims.

## Current implementation gaps found during review

- The existing Python and Kotlin tokenizers differ: Python uses tokenize for valid Python; Kotlin treats // as a comment regardless of language and has different operator/number boundaries and keyword tables.
- The old binary is a feature dictionary with dense float32 class rows, not a neural recurrent tensor container.
- Python training annotations currently use codepoint offsets while predictions use UTF-16. The new schemas require explicit UTF-16 conversion and validation.
- The current ANSI renderer searches for span text; this can select an earlier repeated occurrence. Render by validated coordinates.
- The prototype has no nested region model, scope/reference model, completion model, document revisions, or cross-runtime golden suite.

These are migration tasks. This document does not claim they have been implemented.
