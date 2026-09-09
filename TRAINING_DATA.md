# Multilingual training data requirements

Status: collection and training specification for [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md). No corpus has been collected and no multilingual training is implied by this document.

## Scope and coverage accounting

Use the September 2026 TIOBE top-40 target registry defined in the architecture. Collect code for most applicable textual entries, including ranks 21–40; initial milestones prioritize the top 20. Track parent families, dialects, aliases, exclusions, and supplemental host formats explicitly. The broad-release target is at least 32/40 ranked entries passing declared per-task gates, subject to the registry's no-double-counting rules.

Every canonical language/dialect has separate progress fields. A broad all-three-model support claim requires passing each applicable task gate; publish separate detection/syntax/link/completion counts for partial coverage:

| Field | Evidence required |
| --- | --- |
| Source collected | Accepted, deduplicated source with provenance and language labels |
| Recognition trained/validated | File, snippet and prefix language examples; held-out detection metrics |
| Model 1 trained/validated | Lexical spans and nested region annotations; held-out span metrics |
| Model 2 trained/validated | Identifier/definition/usage/link supervision; held-out exact link metrics and coverage |
| Model 3 trained/validated | Raw ordered source plus word boundaries and causal language targets; held-out completion metrics |
| Mixed-source trained/validated | Language-region boundaries and task labels inside representative embedded regions |

Keep statuses such as PLANNED, COLLECTED, TRAINED, VALIDATED, PARTIAL, and UNSUPPORTED. Store a reason for missing supervision. Raw source used for language detection or completion does not mean semantic linking has been trained. Recognition of a parent family does not establish discrimination between all its dialects.

The Kotlin app must be able to query the bundle's capability metadata and display useful partial results even where some tasks are unvalidated. It must not require the user to identify the language first.

## Where to obtain source

| Source | Intended use | Collection guidance |
| --- | --- | --- |
| Selected public repositories | Main source for all three models | Stream complete project snapshots when semantic indexing needs dependency manifests/build configuration. Record commit and file identity; do not retain the source snapshot after processing. |
| User-owned or otherwise authorized projects | Domain-specific adaptation | Use only projects intentionally included in the corpus; retain an independent general evaluation set. |
| The Stack / The Stack v2 language subsets | Broader raw-code coverage | Filter by canonical language/dialect, provenance, license metadata, duplicates, generated/vendor status, and content quality. These datasets do not directly provide our required span/link labels. |
| Compiler/parser/indexer test suites | Rare syntax, ambiguous constructs, invalid code | Keep expected-valid and expected-invalid tests distinct. Audit suitability and avoid letting one suite dominate training. |
| Controlled generated examples | Shadowing, nesting, dialect contrasts and editor errors | Keep generation provenance, validate labels, and combine with real projects. Do not use them as the entire evaluation set. |

[The Stack v2 dataset card](https://huggingface.co/datasets/bigcode/the-stack-v2/blob/main/README.md) provides language/provenance metadata and describes access and original-license conditions, including agreement requirements for bulk download. Record accepted terms, source licenses and attribution information; missing license metadata is not permission. Preserve removal/exclusion manifests so later releases can be rebuilt.

A loose-file corpus can supply model 1 and model 3. Model 2 usually needs complete projects and dependency/type/schema context for accurate teacher labels. Do not treat missing dependencies or an indexer failure as proof that every reference is unresolved.

Use language-specific sources to close gaps: scientific packages for R/Julia/Fortran/MATLAB, application projects for JVM/.NET/web languages, database scripts with available schema fixtures for SQL dialects, and documented test programs for languages with smaller public corpora. These are collection priorities, not claims that a particular teacher supports every target.

## Streaming acquisition and source-retention policy

The Python trainer must acquire training examples as streams and use them in memory. Raw source is not a training artifact and must not be written to a corpus directory, Python cache, dataset cache, temporary archive, debug log, exception report, TensorBoard event, or checkpoint. This is a hard requirement for the default trainer mode, not merely a recommended cleanup step.

Each source provider exposes an iterator of bounded examples. A provider may read an HTTP response, repository archive, remote dataset shard, or authorized project export incrementally; it must decompress and decode one bounded file/example at a time, enforce byte and nesting limits, yield the example, and release its buffer after the batch and offline teacher steps finish. Large project-level semantic teachers may receive a bounded in-memory project view or a provider-specific remote query; they must not cause a checkout to be persisted locally.

The trainer may write only model checkpoints/exports, aggregate metrics, and a source-use ledger. It must not enable implicit caches. Cache directories for dataset clients, HTTP bodies, archive downloads, parser outputs, and teacher workspaces are disabled or placed in an explicitly ephemeral memory-backed location; if a required third-party tool cannot satisfy this policy, that provider is unavailable in no-retention mode. A failed or interrupted run may restart acquisition; resumability stores provider cursors and ledger keys, never source bytes.

The ledger records what was used without recording the source itself. At minimum, each accepted, rejected, skipped, and failed example records:

```text
runId, providerId, sourceUri, repositoryCommit, relativePath,
contentSha256, byteCount, encoding, licenseId, canonicalLanguageId,
dialectId, split, teacherVersions, labelCoverage, status, reason, timestamp
```

Hash source while it is in memory, then discard it. Do not place source text, snippets, identifiers, stack traces containing source, or raw annotations in the ledger. URL/repository identities and license metadata are still retained for reproducibility and compliance. The ledger itself is append-only, redacted before publication where necessary, and covered by access controls appropriate to private project metadata.

The run manifest also records provider versions, request parameters, accepted terms, registry hash, sampling seed, split/duplicate decisions, hint perturbation rates, teacher configurations, and the exact model/data outputs. A manifest entry proves that an example was considered or used; it does not authorize a source license or replace attribution obligations. Providers must reject missing/unclear rights according to the configured policy before training.

Tests for every provider must assert that no source-bearing path or cache is created, that ledger records contain no source payload, that buffers are released after consumption, and that a second run can reproduce the same source-use decisions from the manifest plus provider cursors. The no-retention mode must be the mode used for private or restricted sources unless the owner explicitly authorizes storage.

## Required provenance and source fidelity

The normalized example held in memory must carry:

- Registry version/hash, canonical language ID, optional dialect/version and original source label.
- Repository/source URL, commit/content hash, relative path, project ID, license metadata, and acquisition version/date.
- Original file encoding and decoding policy, the decoded source only for the lifetime of the example, and its content hash.
- Split assignment, duplicate-cluster ID, generated/vendor/test indicators, and augmentation ancestry.
- Teacher name/version/configuration, success/partial/error state, annotation confidence, and coverage per output head.
- Document/region language evidence, ambiguity, and host/embedded relationships.
- Explicit offsetEncoding=utf16 and validated inclusive-start/exclusive-end ranges.

Keep whitespace, CRLF, strings, comments, and non-ASCII spelling while the example is being processed. Decode legacy encodings into the editor's Unicode text with recorded policy; do not normalize away distinguishing syntax. Retain a mapping to original bytes in memory when needed for teacher offsets, then discard it with the example. Reject malformed mappings rather than silently shifting annotations.

The training loader converts UTF-16 locations to neural byte/scalar boundaries using one checked mapping. Codepoint, byte, and UTF-16 offsets must never be mixed. Full-file and snippet examples must carry the same original document identity and a snippet base offset.

## Language-recognition supervision

Language identity is supervised output, not a mandatory inference input.

Use repository/build metadata, parser/compiler confirmation, and curated review to label files. Extension is useful evidence but not sufficient, particularly for shared extensions and snippets. Cross-check conflicting labels and quarantine uncertain cases.

Create the following example types for every applicable language:

1. Complete files with host-language labels.
2. Crops of varied lengths, including partial lines, functions, arbitrary interior spans, and file prefixes.
3. Ambiguous snippets with compatible-language sets, family labels, or UNKNOWN; a number or shared expression should not inherit a confidently identifiable language solely from its original filename.
4. Paired confusion examples: C/C++, Java/JavaScript, JavaScript/TypeScript, BASIC variants, SQL dialects, Caml/OCaml, and assembly dialects.
5. Unseen/out-of-registry languages, prose, configuration, empty input, and whitespace-only input as abstention/non-code examples.
6. Mixed-language files and snippets with precise language-region ranges and parents.
7. Identical source with no hint, a correct PREFER hint, an incorrect PREFER hint, and controlled FORCE behavior.

AUTO mode must dominate training and evaluation. Set hint dropout/perturbation rates in the run manifest and report them. Keep a code-only evaluation with filenames, paths, extensions, and comments mentioning the language removed from metadata—not removed from the source text itself. Add identifier/comment-language variations to discourage learning natural-language identity as programming-language identity.

At model 3 training time, language supervision and conditioning must match what a prefix reveals. Prefix labels must not leak future annotations into recurrent input. Full-document detection labels may train models 1/2, but do not become forced language tokens for model 3 AUTO training.

For partial or ambiguous labels, mask losses or use compatible-target/set losses. Do not manufacture a single “correct” dialect when syntax does not distinguish it.

## Model 1 annotations

The implemented SYL2 trainer accepts one JSON object per source in an
annotation-manifest JSONL file. It joins records by `uri` or `sourceId`, then
expects `roleSpans` with inclusive/exclusive UTF-16 `start`/`end` coordinates
and a shared `role` (or `kind`). `contentSha256` is optional but recommended;
when present, the trainer refuses to apply the record to a changed source.
Uncovered bytes are masked for the supervised role loss. The current bootstrap
export supports lexical roles and same-window identifier links; nested region
and language-region labels remain additional schema/training work.

Required targets:

- Complete lexical ranges and roles, including whole strings and single-/multiline comments.
- BIOES segmentation derived from those ranges at Unicode scalar boundaries.
- Nested array/object/string/interpolation ranges and parent relations.
- Complete/incomplete flags and exact open-ended boundaries.
- Document and local language-region targets.
- Optional extension labels/subkinds for constructs not represented by shared base categories.

Produce labels offline using lexers, parsers and query mappings. [Tree-sitter queries](https://tree-sitter.github.io/tree-sitter/using-parsers/queries/index.html) are one useful source of syntax-node captures. Each teacher-to-common-schema mapping must be reviewed per language; highlighting query names alone do not supply all aggregate boundaries or definition semantics.

Examples must cover both positive regions and confusing negatives: arrays versus indexing, object literals versus blocks, punctuation in comments, nested literals, escaped delimiters, macros, and incomplete input. Preserve uncommon syntax and malformed editor states instead of filtering only to trivially parsed files.

A failed parser region is unknown supervision. Keep useful raw code for other losses and mask unreliable labels locally. Missing a role category in a language can mean not applicable, rather than zero-quality training.

## Model 2 annotations

The implemented annotation record uses `definitions` with a stable per-source
`id`, `nameRange`, and optional `kind`, plus `usages` with a range,
`definitionId`, `occurrence`, `kind`, and optional `status`. The trainer
converts these coordinates to UTF-8 byte positions, trains occurrence/kind
heads, and applies candidate cross-entropy to resolved definitions that are in
the same bounded encoder window. Resolved links outside that window are
counted as unevaluated coverage; they are never mislabeled as unresolved.
This is a supervised bootstrap for local navigation, not yet full project
semantic training.

Required targets:

- Identifier boundaries, occurrence role, symbol kind, and read/write/import/alias flags where available.
- Definition name, signature and full declaration ranges.
- Enclosing scope ranges and parent relations where available.
- Usage-to-definition IDs with exact target document/range; explicit multiple-valid-target, known-unresolved and external cases.
- Dependencies/index coverage needed to interpret a missing link.

Use compiler semantic APIs or code indexers as offline teachers. [SCIP](https://sourcegraph.com/docs/code-navigation/writing-an-indexer) represents symbol occurrences useful for definition/reference labels. Verify actual teacher support per language and construct. Combine syntax annotations for full declaration ranges with semantic links; SCIP output alone may not provide every scope/signature target.

For languages lacking a usable semantic teacher, obtain a reviewed annotated subset and programmatically validated examples. Do not substitute nearest-name matching as ground truth. Continue collecting raw source and recognition/completion data, but report semantic supervision as missing/partial until true links exist.

Construct hard negatives from same-spelling definitions in different scopes, members on different receivers, aliases, overloads, shadowing and forward references. Train with predicted candidate spans as well as gold ones. Avoid sampling “negatives” that are actually valid alternatives omitted by the teacher.

Single-document local binding is the first cross-language milestone. Imports, cross-project symbols, SQL schema references, dynamic dispatch and cross-language bridges require separate coverage reports. In a mixed-language document, do not link names across languages without explicit semantic evidence.

## Model 3 annotations

Raw source supplies next-byte targets. Additional offline labels supply word/token endings and causal language evidence. Preserve source order and whitespace; never concatenate unrelated files as one prefix without a reset boundary.

Use sequential chunks from each file with carried state and detached gradients for truncated backpropagation. Reset at document boundaries. Record chunk lengths, sampling scheme, state-reset policy, optimizer settings and generation/word-boundary settings.

Include long-prefix examples, partially typed identifiers, comments and literals, host/embedded transitions, and prefixes from every supported dialect. Pairs with the same recent suffix but different earlier code help test whether training uses more than local context.

Word-ending labels must specify handling of whitespace, punctuation, identifier suffixes and valid UTF-8 completion. A teacher failure masks only the auxiliary boundary loss; raw next-byte training can still use the accepted source.

## Mixed-language and editor-state examples

Train on real host/embedded combinations, supplemented by checked compositions: HTML/JavaScript/CSS, PHP templates, SQL strings, Markdown code fences, and shell invocations. Some host formats are supplemental rather than TIOBE-ranked languages.

Annotate embedded code only when warranted. Ordinary prose in a string is not a new program. Track host regions, children and interpolation transitions in original UTF-16 coordinates. Generated compositions must avoid accidental delimiter/escaping errors, or explicitly label those errors as incomplete input.

Generate editor snapshots by cutting inside strings/comments/identifiers, removing a closing delimiter, inserting text, and changing text before a usage. Recompute/validate affected labels and original offsets. Never carry a definition link through an edit if its semantics changed.

Distinguish an identifiable-but-invalid program from an unknown language. Invalid syntax does not necessarily erase language evidence. Keep all augmented descendants of a project in its original split.

## Sampling, splitting and training schedule

1. Freeze collection manifests, language mappings and content hashes.
2. Remove exact duplicates and near-duplicate clusters before splitting; group forks, copied files and related project snapshots together.
3. Assign train/validation/test by project groups. Only then create snippets, incomplete versions and augmentations.
4. Build batches per language and task. A useful starting policy is p(language) proportional to sqrt(accepted examples), with configured floors/caps and enough real distinct projects per language. Record the unit counted: bytes/windows for raw-code losses, annotated examples for supervised losses.
5. Maintain separate pools for language detection, syntax, semantic links and completion; apply only losses with known labels. Do not oversample a tiny low-resource project indefinitely to meet a nominal language quota.
6. Begin with a multilingual pilot containing different syntax families and label qualities. Expand collection and validation toward the target registry; do not treat the pilot's few languages as the final scope.
7. Train shared multilingual encoders/heads and compare capacity profiles. Use per-language results to decide whether larger shared models or bundled learned adapters are necessary.

Per-language manifests must define minimum unique-project, clean-source, lexical-region, definition/link, and held-out counts before a release is called validated. Choose numeric budgets after the initial data inventory; record the chosen thresholds before looking at final test results. This avoids claiming that an arbitrary byte count alone guarantees quality.

Deduplication and corpus splits must also prevent one task's training projects from contaminating another task's held-out evaluation. Keep tokenizer/language-label calibration and threshold selection on training/validation data only.

## Evaluation and release evidence

Report per-language and per-dialect metrics, macro averages, and weakest-language results. Overall byte-weighted accuracy can hide failure on less-common languages.

| Capability | Required reporting |
| --- | --- |
| Recognition | Top-1/top-k, family/dialect confusion, calibration, abstention coverage and unknown/non-code rejection |
| Mixed languages | Exact language-region F1, host/embedded confusion and incomplete-region behavior |
| Model 1 | Lexical/nested span F1, boundary/region-candidate recall, incomplete-input results |
| Model 2 | Exact link accuracy, wrong-link rate, resolved coverage, retrieval recall and name/declaration range accuracy |
| Model 3 | Bits per byte, exact next-word top-k, long-prefix tests, causal language accuracy |
| End-to-end | Task metrics with detected language versus gold language; AUTO versus hinted inputs |
| Kotlin parity | Byte IDs, UTF-16 offsets, language outputs, decoded links/spans and float32 score tolerances |
| Runtime | Model size, peak memory, cold full-prefix and warm incremental latency on declared hardware |

A release includes corpus/registry hashes, per-task data counts, teacher versions, unsupported constructs, confidence thresholds, training settings, evaluation split identities and results. Rank membership is a collection priority; only these results justify a support claim.

Training tooling may require Python ML libraries, parsers and compilers. The exported Kotlin bundle contains weights, vocabulary/registry, calibration and decoding configuration only. None of those offline annotation tools become requirements of the editor client.
