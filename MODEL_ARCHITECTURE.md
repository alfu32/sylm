# Three-model architecture

Status: revised implementation specification, incorporating multilingual training, automatic programming-language recognition, and the requirement that local models do the analysis while the editor gathers results. The repository currently contains only the legacy perceptron highlighter and its SYLM v1 Kotlin runner. The three providers and neural trainers below are proposed work, not existing functionality.

## Review decisions

The three-task split is sound, but the earlier proposal left several requirements unresolved:

| Issue | Decision |
| --- | --- |
| A required language ID prevents automatic recognition | Every model has a learned language head; document language hints are optional. |
| A small training corpus in a few languages cannot establish broad support | Train and evaluate per task across a versioned TIOBE top-40 target registry. |
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

The editor supplies an immutable source snapshot, document identity/revision, and cursor or snippet range. Language is detected automatically; an optional user hint can prefer or force a language. It receives detected-language candidates/regions, lexical spans, nested regions, definitions, usages with target links, and completion candidates. It does not tokenize, classify, parse scopes, rank reference targets, or maintain an analysis symbol table.

The bundled Kotlin inference library performs tensor operations and generic decoding. Learned predictions drive language-sensitive analysis. The library also validates ranges, maps positions, caches document results, and rejects stale results.

~~~text
Python: corpus → train three neural models → export three self-contained binaries
Editor: snapshot → LocalCodeModels Kotlin library → ranges, links, completions
                              |
                   model weights + generic inference
~~~

A network does not itself return editor objects: a decoder converts scores into ranges, links, and strings. That decoder belongs inside the library. There is no required handwritten language grammar, keyword classifier, or deterministic scope resolver in the first model-driven implementation.

Definition links are predictions. Model 2 can abstain or report ambiguity, but cannot promise compiler-equivalent navigation for every language. An optional compiler verification backend can be added later without changing the editor contract.

## Multilingual scope and language registry

Interpret the requested coverage as most languages in the TIOBE top 40, with the top 20 prioritized and ranks 21–40 explicitly included. Kotlin is the implementation language of the client, not a restriction on source languages.

Use a frozen **September 2026** target snapshot, consulted **2026-09-09**. TIOBE updates monthly and describes popularity, not training-data availability or model quality; positions 21–50 are its unofficial extended list. Record the source month with every corpus/model release. The target membership below comes from the [TIOBE Index](https://www.tiobe.com/tiobe-index/).

| Snapshot positions | Target entries, in rank order |
| --- | --- |
| 1–10 | Python, C, C++, Java, C#, JavaScript, Visual Basic, SQL, R, Rust |
| 11–20 | Fortran, Go, Delphi/Object Pascal, PHP, Scratch, Assembly language, Ada, Swift, Objective-C, COBOL |
| 21–30 | Julia, Ruby, Perl, SAS, Classic Visual Basic, Kotlin, MATLAB, Caml, Prolog, GML |
| 31–40 | Lua, PowerShell, D, PL/SQL, ABAP, Transact-SQL, VBScript, OCaml, TypeScript, Zig |

The engineering target is data collection for every applicable textual entry, with a broad-release goal of at least 32 of the 40 TIOBE entries passing the declared per-task gates. JSON, XML, HTML, JSX, TSX, and Svelte are tracked as supplemental host/markup entries and are reported separately from the TIOBE denominator. This is a planned acceptance target, not current support. Publish the exact numerator/denominator and exclusions; do not count a language as link-capable merely because its raw files were used for completion training.

Maintain a stable registry with canonical ID, display name, family, dialect IDs, aliases, TIOBE snapshot entries, input representation, per-task annotation availability, and measured coverage. IDs are never derived from current rank. Registry changes require a new hash and artifact metadata.

Handle representation and ambiguity explicitly:

- Keep Visual Basic (.NET), Classic Visual Basic, and VBScript distinct; do not merge all BASIC-like source into one class.
- Model SQL as a family with generic/unknown dialect and specific dialect IDs, including PL/SQL and Transact-SQL. Missing schema information limits definition linking.
- Assembly requires architecture and syntax/dialect coverage; one assembly sample is not evidence of support for every dialect.
- Record the relationship between Caml and OCaml entries. Parent-family recognition cannot be counted as two independently recognized leaf languages; preserve uncertainty when only the family is identifiable.
- Scratch is deferred from the text-source task until a separate block/serialization representation and location map are defined. Classifying project JSON as JSON would not satisfy Scratch code understanding.
- Include JSON, HTML, CSS, Markdown, and shell snippets as supplemental training context for mixed documents. They do not replace top-40 entries in coverage accounting.
- Language versions, preprocessing modes, source formats (for example fixed/free form), and dialects are metadata and test dimensions. Case sensitivity and syntax conventions are learned from diverse annotations; never globally lowercase source.

### Automatic recognition without a fourth model

All three binaries remain self-contained multilingual models, each with a language head. A request does not require model 1 to run before models 2 or 3.

The registry defines L language outcomes (supported leaf/dialect IDs plus UNKNOWN and NON_CODE), and family membership separately. Ambiguous snippets such as a number, an empty file, or syntax shared by C/C++ must allow multiple candidates or abstention. MIXED is a document composition status, not a single language ID. Similar syntax is not enough to guarantee exact language identification.

For models 1 and 2:

1. Encode source bytes without language hints. AUTO is the default task-conditioning mode; user hints are applied after detection, so detected evidence is independent of the hint.
2. Pool contextual vectors (mean and max: 2D) and apply a learned [L,2D] document-language head. A second [L,D] head gives local language scores; boundary/pair heads decode language regions.
3. Derive a 32-dimensional language vector from local language probabilities and a learned [L,32] embedding table. Fuse it into task features with a learned [D,D+32] projection and tanh. Document probabilities supply a calibrated fallback where local evidence is weak; retain both distributions for diagnostics.
4. The task heads consume language-conditioned vectors of the original width D. There is no circular requirement to know the language before encoding it.

For model 3, predict language from its current causal recurrent state only. Its output conditioning is described in the completion section. Do not call a bidirectional detector on the full editor buffer to select a completion language.

Training combines whole-file, cropped-snippet, and mixed-source language labels with task-specific labels. Explicit hint controls are AUTO, PREFER, and FORCE. AUTO uses detected probabilities; PREFER blends them with a hint prior using a configured weight; FORCE uses the explicit language's embedding for task conditioning. Train predominantly in AUTO mode with hint dropout and occasionally incorrect PREFER hints. Always return independent detected evidence and a conflict flag. File extension/path may supply an optional PREFER prior through the library, never required ground truth at inference.

Calibrate language probabilities on held-out projects, including languages outside the registry and non-code text. Export confidence/margin/entropy thresholds. UNKNOWN is a learned and thresholded abstention; it cannot perfectly detect every unseen language.

### Mixed-language source

Predict nested LanguageRegion ranges with document UTF-16 coordinates, parent IDs, candidates, and confidence. Host and embedded language are separate facts: for example an HTML document containing JavaScript, or SQL embedded in a host-language string. A snippet may have no identifiable host.

Use learned language boundary/pair heads, not client-side grammar switching. The generic decoder enforces containment and preserves original offsets. A string is not automatically an embedded program: train negative examples containing prose, URLs, or code-like fragments.

Expose dominantLanguage only when meaningful; report MIXED or AMBIGUOUS otherwise. Role spans/occurrences refer to their language region. Model 2 defaults to links within a compatible language context and learns compatibility as a feature; cross-language bindings need explicit training labels and must not be inferred by matching names alone.

Model 3 learns host/embedded transitions from prefix examples. Its current-language probabilities can change as bytes arrive. Do not reset its recurrent state on a detected language transition, and do not strip host text from the prefix.


## Shared source and runtime contract

### Original document and snippets

All public ranges and new training annotations use UTF-16 offsets, inclusive start and exclusive end. Offsets index the unchanged source: no newline normalization, whitespace stripping, or Unicode normalization. Line and column are zero-based; columns use UTF-16 units. An LSP adapter must honor the position encoding negotiated with its client.

~~~kotlin
data class DocumentId(val uri: String, val revision: Long)
data class SourceRange(val start: Int, val end: Int)
enum class LanguageHintMode { PREFER, FORCE }
data class LanguageHint(
    val languageId: String,
    val mode: LanguageHintMode = LanguageHintMode.PREFER,
)
data class DocumentSnapshot(
    val id: DocumentId,
    val text: String,
    val languageHint: LanguageHint? = null, // null = AUTO
)
data class SourceLocation(val document: DocumentId, val range: SourceRange)
~~~

Build Python codepoint-to-UTF-16 and UTF-8-byte-to-UTF-16 maps once per snapshot. Never locate an occurrence by searching for its spelling. Validate bounds and scalar boundaries. A range must not split a surrogate pair. For a temporarily unpaired surrogate in an editor buffer, use a documented replacement-byte view for inference while retaining its original UTF-16 position.

A snippet carries its original DocumentId and baseOffsetUtf16. Convert local [s,e) to original [base+s,base+e). Verify that the snippet is the corresponding substring of that revision. A detached snippet can give valid locations but has limited surrounding context; report that limitation in the result. Prefer analyzing a document snapshot and querying spans intersecting the snippet. Do not clip a complete comment or array into a falsely complete region.

Results carry the input revision. After an edit, stale results must be discarded or explicitly remapped by the editor. Source text is obtained from the matching snapshot; repeated copies of large nested substrings are optional convenience values, not stored in every span.

### Shared lossless input

All three models consume UTF-8 bytes of the original source plus BOS. Language recognition uses only source; optional hints affect the subsequent language-conditioning layer. Preserve whitespace, line breaks, comments, and literal contents. No language lexer, keyword lists, Python tokenize call, or normalization participates in neural inference.

A portable, language-independent encoder maps bytes back to UTF-16 ranges. Structural heads operate only at Unicode scalar boundaries; continuation-byte positions are masked during span decoding. Invalid intermediate editor text follows the replacement-byte rule above.

The input vocabulary consists of 256 bytes plus BOS, EOS and PAD (V=259 in the reference profile). The output language registry and its conditioning embeddings are separate from this byte-input vocabulary. Unsupported hints are rejected explicitly or mapped to AUTO with a diagnostic. Exact IDs and hashes are exported and tested across Python and Kotlin.

### Language result contract

~~~kotlin
enum class LanguageStatus { DETECTED, AMBIGUOUS, MIXED, UNKNOWN, NON_CODE }
enum class LanguageEvidenceScope { DOCUMENT, AT_CURSOR }
data class LanguageCandidate(
    val languageId: String,
    val probability: Float,
)
data class LanguageRegion(
    val id: Int,
    val parentId: Int?,
    val location: SourceLocation,
    val candidates: List<LanguageCandidate>,
)
data class LanguageDetection(
    val document: DocumentId,
    val scope: LanguageEvidenceScope,
    val status: LanguageStatus,
    val candidates: List<LanguageCandidate>,
    val regions: List<LanguageRegion>,
    val analyzedRange: SourceRange,
    val hintConflict: Boolean,
    val registryHash: String,
)
~~~

Candidates are ranked and calibrated; an output top-k list need not sum to one because omitted outcomes retain probability mass. Document revision and analyzedRange describe the evidence used. DOCUMENT results describe overall language/composition and detected regions. AT_CURSOR results describe the current language from a prefix; analyzedRange ends at the cursor and regions may be empty because full nested language-region analysis was not requested. An AT_CURSOR status does not assert that the entire document uses that language. Hints and effective task conditioning are reported separately from detected evidence.

### Common neural encoder for models 1 and 2

Share implementation and architecture, initially with independent weights per artifact:

| Layer | Proposed shape | Purpose |
| --- | --- | --- |
| Byte/special-token embedding | [V,64] | Encodes lossless source, including arbitrary identifier spellings. |
| Context GRU layer 1 | bidirectional, input 64, H=64 per direction | Learns spelling, delimiters, lexical modes, and syntax from both directions. |
| Context GRU layer 2 | bidirectional, input 128, H=64 per direction | Produces a 128-vector at each source byte using wider context. |
| Language heads and fusion | document [L,256], local [L,128], language embedding [L,32], fusion [128,160] | Recognize source language and condition task features while keeping output width 128. |
| Task heads | described below | Predict lexical spans, nested regions, declarations, and links. |

For each decoded interval, its representation is the mean of the 128-vectors inside it. Prefix sums make pooling efficient. Source coordinates remain outside the learned network and are never approximated by it.

Process the whole analysis unit initially. Long documents can be processed in chunks while carrying forward/backward states across chunk boundaries; reset at document boundaries, not arbitrary chunk boundaries. A bounded-context optimization must expose its coverage and be evaluated separately.

These dimensions are a compact reference configuration, not a claim that this capacity suffices for 30–40 languages. Keep dimensions in metadata. Compare H=64/128/256 encoder configurations using per-language metrics before selecting a deployable model. Language imbalance and interference may require larger shared weights or learned per-language adapters, bundled inside the same artifact. Byte input trades a simple portable tokenizer for longer sequences. Source length affects work and activation memory. Weights are immutable and reusable; request/session state is private to each document analysis.

## Model 1: lexical roles and nested source regions

### Outputs and label semantics

The requested categories are retained:

~~~text
KEYWORD, IDENTIFIER, PUNCTUATION, STRING_LITERAL, NUMBER_LITERAL,
ARRAY_LITERAL, OBJECT_LITERAL, SINGLE_LINE_COMMENT, MULTILINE_COMMENT
~~~

Add OPERATOR, BOOLEAN_LITERAL, NULL_LITERAL, and UNKNOWN to avoid forcing ordinary source into a wrong requested category. The annotation specification defines literal semantics for each language; the model learns them from labeled examples. A Python list may map to ARRAY_LITERAL and a dictionary to OBJECT_LITERAL; sets, blocks, indexing, and constructor calls must not be labeled as those literals just because they contain brackets or construct a collection.

Lexical/region labels are shared across languages, with optional language-specific subkind metadata. Distinct constructs (sets, tuples, preprocessor directives, labels, annotations, etc.) must receive an explicit extension label or UNKNOWN; do not force them into an unrelated universal category. Label vocabularies and language applicability are versioned.

There are two structural outputs in addition to language recognition:

- Lexical spans: ordered, non-overlapping outer lexemes with exact boundaries. Adjacent equal labels are not automatically merged.
- Region spans: nested structural ranges with parent IDs, including arrays, objects, and interpolated expressions/literals where necessary.

Interpolated strings keep an outer STRING_LITERAL range and nested expression spans, preserving identifiers used inside them.

A complete comment covers its delimiters and content. A single-line comment excludes its terminating newline; a block comment uses MULTILINE_COMMENT even if its text happens to fit on one line. An unterminated comment/string extends to the available input boundary with complete=false.

### Layers and decoding

1. **Lossless input and embedding:** encode bytes with the common embedding table. The network sees comment delimiters, spaces, quote characters, and names directly.
2. **Bidirectional context and language heads:** produce a 128-vector per byte, document/local language probabilities, and language-conditioned task vectors. Both sides of a position may be used for document analysis.
3. **Lexical segmentation head:** linear [4×Clex+1,128] plus bias predicts BIOES role tags at Unicode scalar positions: beginning, inside, end, single, plus outside/trivia. Clex is the exported lexical-role count (11 for the proposed roles, excluding arrays/objects). A constrained sequence decoder enforces valid transitions and combines complete multiword/multiline lexemes. A separate [1,128] completeness head scores pooled spans; an open sequence at the document end may decode as incomplete rather than inventing a closing delimiter.
4. **Region boundary heads:** independent projections score starts, ends, type, and completeness for ARRAY_LITERAL, OBJECT_LITERAL, STRING_LITERAL, and INTERPOLATION_EXPRESSION. Keep multiple candidate boundaries to support nesting.
5. **Region pairing head:** concatenate opening, closing, and mean interior representations (3×128) plus 16 generic length/boundary features. Apply linear [64,400], tanh, linear [5,64] plus biases. Outcomes are the four region kinds plus NONE. Missing ends use a virtual document-end candidate with a completeness flag. Learn pairing rather than infer it from bracket counting.
6. **Generic span decoder:** select compatible scored regions, reject crossing ranges, and derive parent IDs by containment. Candidate pruning and beam limits are artifact configuration; report incomplete/pruned coverage. The head cannot recover a region whose boundary was discarded.

Language meaning comes from annotations and learned scores. Decoding enforces coordinate and nesting invariants only. A flat BIOES output covers outer lexemes; region heads expose interpolation expressions, and model 2 independently detects identifier occurrences within them.

~~~kotlin
data class LexicalSpan(
    val id: Int,
    val languageRegionId: Int?,
    val location: SourceLocation,
    val role: String, // validated against artifact labels
    val complete: Boolean,
)
data class RegionSpan(
    val id: Int,
    val languageRegionId: Int?,
    val parentId: Int?,
    val location: SourceLocation,
    val role: String,
    val complete: Boolean,
)
data class SourceRoleResult(
    val document: DocumentId,
    val language: LanguageDetection,
    val lexical: List<LexicalSpan>,
    val regions: List<RegionSpan>,
    val contextComplete: Boolean,
    val candidateSearchLimited: Boolean,
)
~~~

Language region IDs refer to the attached LanguageDetection; null means language context is unknown or document-wide. Parent IDs are result-local. Display styling prefers a lexical child over its enclosing aggregate region; highlighting an entire array must not obscure its identifiers or comments.

### Training and evaluation

Use lexer/parser-derived annotations offline plus explicit incomplete-source examples. These tools are training dependencies only. Keep lexical labels and regions separate. Loss combines document/local language losses, lexical tag cross-entropy, boundary/completeness loss, and region pair classification, masking unannotated positions. Include negative boundary pairs and block/indexing examples. The example below is partial; omitted labels are not silently assigned a heuristic ground truth.

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

Declaration, scope, visibility, aliasing, and usage relationships are learned from semantic annotations. The Kotlin client has no deterministic scope resolver. Single-document links across a multilingual pilot set are the first delivery slice. Data collection targets local links for the entire textual target registry; complete typed/import/overload coverage is reported separately. Cross-file links require other documents or a versioned library-owned index as input; weights cannot know definitions in files never supplied.

### Layers

1. **Contextual encoder and language heads:** independently recognize document/local language and produce language-conditioned 128-vectors. Model 2 can run without model 1.
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
    val languageId: String?,
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
    val languageId: String?, // null if unresolved
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

Obtain identifier, declaration, scope, occurrence, and reference-link annotations offline from compiler/semantic tooling, supplemented by checked annotations. A syntax tree alone cannot label all reference targets. Train boundary/role heads with classification losses, endpoint heads with pointer losses, and the link head with candidate cross-entropy including null/multiple-valid-target cases. Mask unavailable labels per task and language. Include language classification/region losses and same-name inner/outer definitions as hard negatives. Do not mark unsupported semantic cases as known-unresolved ground truth. Train/evaluate with predicted as well as gold candidates to expose cascading errors.

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

Capacity experiment: the Python trainer now supports configurable embedding
width, GRU width, and GRU depth. The original 96/192/2 configuration contains
473,298 parameters with the current 47-language registry. The larger experiment
uses 160-dimensional embeddings and three 384-unit causal GRU layers, totaling
2,562,322 parameters. Each layer updates a recurrent summary of the prefix;
the final layer feeds the byte, boundary, and language heads. Its recurrent
state is 1,152 float32 values (4,608 bytes), excluding weights and checkpoints.
`completionConfig` in the SYL2 export records actual dimensions and parameter
count. This requires a dimension-aware SYL2 runtime; the legacy Kotlin SYLM1
runner cannot execute these neural artifacts. The tables below describe the
original baseline; this experiment scales capacity without changing its
causal byte-input contract. Validation currently reports next-byte top-1
accuracy, despite the legacy `precision` field name.

### Selected model and meaning of context

Use a two-layer unidirectional GRU. Feed every byte of the source before the cursor in order, carrying hidden state throughout the prefix. There is no default left truncation or fixed n-gram window.

The hidden state is a finite learned summary: all earlier bytes can influence a prediction, but exact recall of arbitrarily distant code is not guaranteed. A GRU is suitable for streaming the whole prefix with bounded recurrent state. A conventional finite-window Transformer would change the context contract and is not the initial implementation.

### Layers

| Layer | Proposed shape | Purpose |
| --- | --- | --- |
| Lossless byte tokenizer | 256 UTF-8 byte IDs plus BOS, EOS, PAD | Retains spaces/newlines and represents unseen identifiers. Special IDs are fixed by artifact metadata. |
| Embedding | [V,96], V=259 | Maps each input byte/special ID to a learned vector. |
| Causal GRU layer 1 | input 96, hidden 192 | Updates prefix state from each embedding. |
| Causal GRU layer 2 | input 192, hidden 192 | Builds a second recurrent representation. |
| Causal language head | [L,192] plus bias [L] | Detects the current source language from prefix state only. |
| Language conditioning | [L,32] language embedding; fusion [192,224] plus bias and tanh | Combines the 192-vector with a 32-vector derived from language probabilities or an explicit hint. |
| Vocabulary projection | [257,192] plus bias [257] | Predicts a next byte or EOS from the conditioned vector; BOS and PAD are never generated. |
| Word decoder | beam search plus learned word-boundary head | Accumulates generated bytes into a word/token or identifier suffix. |

Feed BOS before source bytes. The causal language head reads the unconditioned second-layer state; language embeddings condition only output heads, avoiding a circular encoder dependency. AUTO uses detected probabilities, PREFER blends a hint prior, and FORCE follows an explicit hint; detected evidence remains source-only in all modes. Reset state between documents, not automatic language switches. GRU order is recurrent, so no positional embedding or attention KV cache is used. The two hidden vectors require 384 float32 values (1,536 bytes); weights, document text, editor checkpoints, and beam states require additional memory.

For this reference configuration the initial unquantized model has 482,370 + 96×V + 225×L parameters including byte/special embeddings, two GRUs, the language head/embedding/fusion, output projection, and a [1,192] word-boundary head with bias. V is the input vocabulary size (259) and L the language-outcome count. Wider multilingual profiles require recalculation from exported tensor shapes. This is a sizing calculation, not a performance result. Measure cold prefix-processing latency and warm next-word latency separately.

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

The neural objective is next-byte prediction. The user-facing provider returns completed words/tokens. The language and word-boundary heads are trained on all available target languages; optional dialect hints cannot replace AUTO-mode evaluation:

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
    val language: LanguageDetection,
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

Initial correctness can recompute the prefix. Optimization stores hidden-state checkpoints keyed by document revision, registry/model hash, and UTF-16 position. Automatic language probabilities are state-derived and never injected retroactively. Changing a hint invalidates conditioned predictions/beam caches but can reuse unconditioned prefix state. After insertion/deletion/replacement, find a verified unchanged prefix, resume before the edit, and replay bytes through the cursor. Cursor movement backward must not expose future state. Bound checkpoint memory independently of document size.

If a large-prefix computation is cancelled or unfinished, return pending/incomplete context rather than claim the whole prefix was used. Cold full-prefix processing is linear in prefix length; there is no constant-latency guarantee.

### Training and evaluation

Train with next-byte cross-entropy, word-boundary binary cross-entropy, causal language loss, and truncated backpropagation through time. A prefix receives a definitive language target only when its evidence supports one; use soft/family/unknown targets for ambiguous prefixes. Never condition the training input on a future-derived language annotation that AUTO inference cannot access. Carry the numerical hidden state between consecutive chunks from the same file while detaching gradients at chunk boundaries. Reset only at file boundaries; shuffled unrelated chunks cannot share state. Gradient truncation is a training limitation, distinct from discarding prefix input at inference.

Use project-separated, deduplicated code corpora with recorded provenance. Report held-out bits per byte and exact next-word top-k after decoding. Test long-distance examples with identical recent suffixes but different early declarations, and compare to an n-gram baseline. Compare fresh full-prefix inference against append and mid-document replay. Processing the entire prefix is a structural guarantee; successfully using distant information is an empirical property.

## Kotlin facade: application gathers results

The facade owns model loading, inference, result indexes, and per-document caches. A minimal API is:

~~~kotlin
data class SymbolAnalysis(
    val document: DocumentId,
    val language: LanguageDetection,
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
    fun detectLanguage(snapshot: DocumentSnapshot): LanguageDetection
    fun analyze(snapshot: DocumentSnapshot): CodeAnalysis
    fun definitionAt(snapshot: DocumentSnapshot, offsetUtf16: Int): DefinitionQueryResult?
    fun complete(snapshot: DocumentSnapshot, cursorUtf16: Int, topK: Int = 8): CompletionResult
    fun invalidate(documentUri: String)
}
~~~

The editor loads the three binaries once, submits snapshots on a background worker, and applies results only when revisions match. Highlighting uses roles; clicking an identifier uses definitionAt; a definition preview slices the matching snapshot using declarationRange; typing uses complete. The app supplies snapshots for any external documents it wants analyzed, but the library owns their analysis index.

The facade uses model 1 for explicit full-document detectLanguage queries; models 2 and 3 retain their own language heads. Independent heads can disagree: report per-task evidence, use calibrated uncertainty, and avoid hiding a disagreement behind a forced global ID. Completion results use only their analyzed prefix.

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

Metadata includes task ID, architecture/operator version, tensor names/shapes, the shared language registry/hash and TIOBE snapshot, per-language per-task trained/validated/unsupported status, language-head calibration and hint policy, byte-encoding and output-decoding versions, vocabulary/labels, output head schemas, normalization rules, thresholds, and training provenance. Each artifact includes its required encoding/decoding configuration. The three artifacts must have compatible language/role schemas; reject incompatible bundles instead of silently remapping IDs. Their learned weights remain independent; Kotlin needs no adjacent Python files or training checkpoint.

The loader validates task, architecture, dimensions, byte counts, duplicate names, finite values, checksum, and allocation limits before using tensors. Explicitly define metadata fields and fixtures before implementing loaders. Do not load Python pickle/Torch checkpoints in Kotlin. Session states are runtime values, not model weights exported from training.

Python's reference inference must reload exported float32 artifacts for comparison; comparing Kotlin against unexported float64 training weights hides export drift. Pin a tested PyTorch training dependency when implementing the trainer; the current standard-library-only requirements file describes only the legacy CLI.

## Multilingual training requirements

The detailed corpus/annotation requirements and collection guidance are in [TRAINING_DATA.md](TRAINING_DATA.md). They are part of this specification. Key requirements:

- Collect representative source for the versioned top-40 registry, with explicit exceptions and dialect mappings. Keep language recognition, role/region annotations, semantic links, and completion coverage separate.
- Train all models in AUTO mode as the default, using supported-language, unknown-language, non-code, short-prefix, and mixed-source examples. A code file's project label is not automatically an identifiable label for every crop.
- Balance language exposure using capped temperature sampling and per-language quotas. Compute semantic-head batches only from actual semantic annotations; mask absent labels rather than teach false negatives.
- Generate parser/compiler/indexer annotations offline; Kotlin has no dependency on annotation tools. Code-only pretraining does not establish definition-link supervision.
- Acquire training data through bounded streaming providers: use source in memory, discard it after the batch/teacher step, and write only model artifacts, aggregate metrics, and a redacted source-use ledger. No raw source or implicit dataset/archive/parser cache is allowed in the default trainer mode; see [TRAINING_DATA.md](TRAINING_DATA.md).
- Split and deduplicate by project before cropping or augmentation. Evaluate macro averages, worst-language results, dialect/confusion groups, calibration, and per-task coverage as well as overall totals.
- Scale model width only with measured benefit and device budgets. The old tiny bootstrap corpus and reference parameter sizes do not demonstrate broad multilingual capability.

## Implementation order and acceptance gates

1. Freeze the language registry, TIOBE snapshot and corpus manifest; collect a multilingual pilot. The legacy `train-stream` command is the initial bounded raw-file acquisition/ledger slice. Implement shared snapshot/range types, byte-encoding specification, SYL2 schema, and Python/Kotlin fixtures. Start neural work only after matching byte IDs and coordinates.
2. Implement the Python trainer/exporter, language heads/conditioning, and Kotlin GRU/linear operators. Compare single-step and sequence states, reverse-direction outputs, and exported logits.
3. Deliver model 1 lexical and nested-region training/inference, with exact offset tests and parser/scanner baselines.
4. Deliver model 2 identifier, declaration/scope, and link training/inference. Declare validated semantic coverage per language and publish incorrect-link rates.
5. Deliver the causal GRU completion trainer and Kotlin word decoder with full-prefix replay; an n-gram benchmark does not fulfill this delivery.
6. Expand training toward the textual top-40 target; satisfy the declared per-task/per-language gates. Package three multilingual providers, automatic-language examples, model metadata, and held-out reports. A pilot release is explicitly labeled with its narrower coverage.

Cross-runtime tests must cover language probabilities and region IDs in AUTO/hint modes, mixed-language boundaries, causal detection without suffix leakage, comments, triple/raw/interpolated strings, operators, Unicode including emoji before a range, CRLF, nested arrays/objects, incomplete input, repeated spellings, and edits. Require exact tokenizer IDs/ranges and decoded structure, with documented float32 numerical tolerances for logits. Near ties must be identified instead of asserting bit-identical probabilities.

Model artifacts and nine bootstrap snippets alone do not establish editor quality. Report language macro-F1/top-k, abstention precision/coverage, unknown rejection, mixed-region F1, task metrics conditioned on both gold and detected language, model sizes, device/JVM, peak memory, and cold/warm latency before making quality/performance claims.

## Current implementation gaps found during review

- The existing Python and Kotlin tokenizers differ: Python uses tokenize for valid Python; Kotlin treats // as a comment regardless of language and has different operator/number boundaries and keyword tables.
- The old binary is a feature dictionary with dense float32 class rows, not a neural recurrent tensor container.
- `train-stream` supports bounded raw files/HTTP(S) files and legacy bootstrap labels. It does not yet stream repository archives/dataset shards through project-level semantic teachers.
- `training.sylm1_trainer` and `training.syl2_trainer` provide streaming trainers. SYL2 can consume a separate gold annotation manifest: role spans and local definition links are trained when present, while completion remains self-supervised. Missing labels are masked; they are not converted into false negatives.
- The supervised annotation schema uses the public UTF-16 coordinate contract and the trainer validates/converts ranges to UTF-8 byte positions. Same-window link training is implemented; cross-window/cross-file semantic coverage, Kotlin SYL2 inference, and broad teacher adapters remain required for a validated release.
- The current ANSI renderer searches for span text; this can select an earlier repeated occurrence. Render by validated coordinates.
- The prototype has no nested region model, scope/reference model, completion model, document revisions, or cross-runtime golden suite.

These are migration tasks. This document does not claim they have been implemented.
