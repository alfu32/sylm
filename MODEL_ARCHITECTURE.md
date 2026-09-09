# Three-model architecture

This document defines the intended architecture for three independent code-intelligence models. Each model has a separate training target, artifact, and Kotlin provider.

~~~text
                         source document
                               |
                   shared source-coordinate layer
                               |
          +--------------------+--------------------+
          |                    |                    |
   Model 1: roles       Model 2: symbols      Model 3: completion
   source spans         definitions/usages   next-word candidates
          |                    |                    |
          +--------------------+--------------------+
                               |
                         Kotlin editor APIs
~~~

The Python CLI trains and exports model artifacts. Kotlin loads those artifacts and runs inference without invoking or importing Python.

## Shared foundation

### Layer 0: source-coordinate map

Input is the original source string, language identifier, and document URI. Every token and result keeps:

- UTF-16 start offset;
- UTF-16 exclusive end offset;
- line and column for editor protocols;
- original source text.

The runtime must never find a range by searching for token text. Repeated identifiers make that unsafe. Ranges come from the tokenizer's position in the original document.

### Layer 1: language-aware lexer

The lexer divides source into significant tokens while preserving gaps. It recognizes identifiers, keywords, punctuation, operators, strings, numbers, comments, and opening/closing delimiters for arrays and objects.

The lexer is part of the standalone Kotlin runtime. It is not a Python dependency. Python and Kotlin must use compatible token boundaries and feature names. Initial language adapters are Python, JavaScript/TypeScript, Kotlin, JSON, and a generic fallback.

### Layer 2: shared token representation

~~~kotlin
data class SourceToken(
    val id: Int,
    val text: String,
    val start: Int,
    val end: Int,
    val line: Int,
    val column: Int,
    val lexicalHint: String,
)
~~~

lexicalHint is an observation from the lexer, not a model prediction. For example, a token can have the hint quoted while model 1 decides the complete region is a string literal.

### Layer 3: artifact boundary

Every artifact contains a format version, task name, vocabulary, model parameters, and feature schema. The Kotlin loader rejects an unknown version or wrong task.

The current SYLM version-1 sparse matrix is a linear-model prototype. It is suitable for small classification heads. Model 3 needs a tensor artifact containing embeddings and recurrent or attention weights; it cannot be represented by one classification matrix.

## Model 1: source-role and literal-region classifier

### Purpose

Model 1 answers: What is this source region? Its results must map directly into the original editor buffer.

The initial role vocabulary is:

~~~text
KEYWORD
IDENTIFIER
PUNCTUATION
STRING_LITERAL
NUMBER_LITERAL
ARRAY_LITERAL
OBJECT_LITERAL
SINGLE_LINE_COMMENT
MULTILINE_COMMENT
~~~

Comments and literals are spans, not words. A multiline comment is one MULTILINE_COMMENT region. An array such as [1, name, "x"] is one ARRAY_LITERAL region.

### Pipeline

~~~text
source
  → coordinate map
  → language lexer
  → token/character embeddings
  → context encoder
  → token-role head and region-boundary head
  → span decoder
  → SourceSpan results
~~~

### Layers

#### 1. Input and position layer

Produces token IDs, character/subtoken IDs, lexical hints, delimiter depth, and source positions. Position and delimiter depth help distinguish a declaration from an expression.

#### 2. Token and character embedding layer

Known keywords and punctuation get token embeddings. Unknown identifiers use character or subtoken features so related names share structure. Literal contents can be abbreviated; the original literal remains outside the model for output.

#### 3. Context encoder

Reads nearby tokens and, ideally, both left and right context. Right context is important for deciding whether delimiters enclose an array, object, or another construct. The MVP can use a fixed local window. The target can use a small bidirectional recurrent or attention encoder.

#### 4. Token-role head

Predicts keyword, identifier, punctuation, string-literal, and number-literal roles. BIO or BIOES labels allow regions to cover multiple tokens.

#### 5. Region-boundary head

Predicts opening/closing boundaries and region types for arrays, objects, and comments. It supports nested regions, such as an object containing an array containing a string.

#### 6. Span decoder

Converts token labels and boundaries into ordered, validated source ranges. It returns both offsets and the exact original substring. Parent and child spans may overlap; the editor can choose the most specific style.

### Output contract

~~~kotlin
enum class SourceRole {
    KEYWORD, IDENTIFIER, PUNCTUATION, STRING_LITERAL, NUMBER_LITERAL,
    ARRAY_LITERAL, OBJECT_LITERAL, SINGLE_LINE_COMMENT, MULTILINE_COMMENT
}

data class SourceSpan(
    val start: Int,
    val end: Int,
    val role: SourceRole,
    val text: String,
    val depth: Int = 0,
)
~~~

start is inclusive, end is exclusive, and both are UTF-16 offsets.

### Training data

Training records contain source and complete annotated regions:

~~~json
{
  "language": "javascript",
  "code": "const xs = [1, 2]; /* values */",
  "spans": [
    {"start": 0, "end": 5, "role": "KEYWORD"},
    {"start": 11, "end": 17, "role": "ARRAY_LITERAL"},
    {"start": 19, "end": 31, "role": "MULTILINE_COMMENT"}
  ]
}
~~~

## Model 2: definitions, usages, and symbol links

### Purpose

Model 2 answers which identifiers define symbols, which identifiers use symbols, and which definition each usage refers to. Its result powers go-to-definition, symbol navigation, and references.

### Pipeline

~~~text
source spans and tokens
  → identifier candidate layer
  → contextual symbol encoder
  → definition/usage classifier
  → scope and declaration graph
  → reference resolver
  → editor lookup index
~~~

### Layers

#### 1. Identifier candidate layer

Selects identifier-like tokens and retains exact ranges. Keywords, punctuation, comments, and literals are excluded unless a language adapter says they introduce a name.

#### 2. Contextual symbol encoder

Encodes each candidate with nearby syntax, declaration keywords, delimiters, member access, imports, enclosing function/class, and document/module identity.

#### 3. Definition/usage head

Predicts one of:

~~~text
DEFINITION, USAGE, IMPORT, LABEL, PROPERTY, UNKNOWN
~~~

Definitions also receive a kind such as function, class, variable, parameter, property, or type.

#### 4. Scope and declaration graph

Builds nested module, function, class, and block scopes. Each definition becomes a declaration node. Scope nesting is required for shadowing: an inner count can hide an outer count.

#### 5. Reference resolver

Connects each usage to the best visible definition. Known language rules should be deterministic. Model scores can rank ambiguous candidates. If confidence is insufficient, return no link instead of navigating to the wrong symbol.

#### 6. Editor lookup index

Indexes occurrences by document range and definitions by symbol key. It provides definition-at-cursor, usages-at-cursor, and references-of-definition queries.

### Output contract

~~~kotlin
data class DefinitionLocation(
    val documentUri: String,
    val start: Int,
    val end: Int,
    val name: String,
    val kind: String,
)

data class IdentifierUsage(
    val documentUri: String,
    val start: Int,
    val end: Int,
    val name: String,
    val definition: DefinitionLocation?,
    val confidence: Float,
)
~~~

The editor navigates using documentUri, start, and end. All offsets are UTF-16 and refer to the original document.

Recommended queries:

~~~kotlin
fun definitionAt(documentUri: String, offset: Int): DefinitionLocation?
fun usagesAt(documentUri: String, offset: Int): List<IdentifierUsage>
fun referencesOf(definition: DefinitionLocation): List<IdentifierUsage>
~~~

### Training data

Usage links identify definitions by annotation ID, not by spelling:

~~~json
{
  "documentUri": "file:///workspace/main.kt",
  "code": "fun inc(count: Int) = count + 1",
  "definitions": [
    {"id": "d0", "start": 4, "end": 7, "name": "inc", "kind": "function"},
    {"id": "d1", "start": 8, "end": 13, "name": "count", "kind": "parameter"}
  ],
  "usages": [
    {"start": 22, "end": 27, "name": "count", "definitionId": "d1"}
  ]
}
~~~

## Model 3: causal next-word model

### Purpose

Model 3 predicts the next source word or token from the code prefix. It is a completion model, not a syntax-role classifier.

~~~text
prefix:  "fun inc(x: Int): Int = x +"
outputs: " 1", " x", " 2", ...
~~~

Entire previous code requires model state. A fixed classification matrix cannot provide unbounded memory. The practical target is a causal model with a context window, for example 2K to 8K tokens, and a cached recurrent or attention state. Longer files need an explicit truncation or memory policy.

### Pipeline

~~~text
source prefix
  → code tokenizer
  → token/subtoken embeddings
  → positional encoding or recurrent state
  → causal GRU or Transformer layers
  → vocabulary projection
  → top-k completion policy
  → completion result
~~~

### Layers

#### 1. Code tokenizer

Converts source into words, identifiers, operators, punctuation, and subword pieces. Splitting identifiers lets the model complete unseen names.

#### 2. Embedding layer

Maps token IDs to dense vectors. A language ID and optional editor context can be prepended.

#### 3. Position/state layer

Records order. A Transformer uses positional encoding and a key/value cache. A GRU uses a recurrent hidden state. This state supports incremental editor typing.

#### 4. Causal context layers

Each position sees only earlier positions. A small model can use a few GRU layers or Transformer blocks, selected according to mobile latency and memory limits.

#### 5. Vocabulary head

Projects the final state to vocabulary logits. Top-k selection can operate directly on logits; probabilities are optional.

#### 6. Completion policy

Filters candidates, applies temperature/top-k/top-p when requested, and returns text plus scores. It must not modify the document.

### Output contract

~~~kotlin
data class NextTokenCandidate(
    val text: String,
    val score: Float,
)

data class CompletionResult(
    val candidates: List<NextTokenCandidate>,
    val contextTokensUsed: Int,
    val truncated: Boolean,
)
~~~

Recommended stateful API:

~~~kotlin
interface CodeCompletionModel {
    fun reset(language: String)
    fun append(sourceChunk: String)
    fun predictNext(topK: Int = 8): CompletionResult
}
~~~

### Training data

Training uses code documents split into prefix/next-token pairs with causal next-token cross-entropy. Documents must be split before train/validation/test splitting to avoid leakage.

## Artifacts and Kotlin providers

Keep provider boundaries separate even if low-level tensor code is shared:

~~~kotlin
class SourceRoleProvider(...)
class SymbolRelationProvider(...)
class CodeCompletionProvider(...)
~~~

Suggested artifacts:

~~~text
token-role.model.bin
identifier-relation.model.bin
next-word.model.bin
~~~

The first two can initially use versioned float32 matrices. Model 3 needs embeddings, vocabulary, recurrent or attention weights, cache metadata, and possibly quantization metadata, so it needs a tensor-container format.

## Implementation phases

1. Replace the current role vocabulary with model-1 span labels and add complete array, object, and comment regions.
2. Add model-2 annotations, scope graph, definition links, and editor lookup APIs.
3. Add a bounded-context model-3 prototype and Kotlin top-k inference.
4. Replace the bounded model with a small GRU or causal Transformer and cached incremental inference.
5. Add language adapters and evaluation: span F1, definition-link accuracy, and completion top-k/perplexity.
