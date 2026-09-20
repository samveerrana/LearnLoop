# LearnLoop

LearnLoop is an experimental, reversible learning-and-tool layer for local language models.
The permanent base checkpoint remains recoverable. A session may load temporary LoRA
parameters, use verified local tools and official API retrieval, then discard the session update.

## Current milestone: v0.5 research prototype

- Runs Qwen3-4B locally with Apple MLX
- Saves corrections in an inspectable SQLite database
- Retrieves relevant corrections during later answers
- Automatically researches common factual questions and stores sourced knowledge
- Recalls researched knowledge after a complete restart
- Exports approved examples in MLX-LM chat-training format
- Never lets generated text silently overwrite the base model
- Builds temporary rank-64 parameter capsules with a 5 GiB disk ceiling
- Runs a frozen 100-question benchmark with per-category scores and provenance
- Refuses the V1 claim unless a real approximately-20B reference and every feature gate pass

## Run

```bash
uv sync
uv run learnloop chat
```

For the simplest terminal experience, start Ollama and run one command:

```bash
uv sync --no-editable
.venv/bin/learnloop-start
```

LearnLoop lists locally installed Ollama models and asks which model to use. Its conservative
artifact-size heuristic keeps
a 10 GiB physical-memory reserve, so the 13 GB 20B artifact is refused on this 16 GiB Mac
but can be selected on a sufficiently larger machine. The selected Ollama
model receives LearnLoop's sourced memory, official-API retrieval, corrections, and tools.
Temporary MLX parameter experts use a separate compatible MLX checkpoint. The current
`learnloop-start` Ollama chat does **not** load those MLX experts; it provides sourced memory,
retrieval, corrections, and tools. Expert activation remains a separate MLX workflow.

### Multi-expert parameter budget

Temporary expert packs can total up to 400 million verified tensor parameters on SSD.
The registry/router selects at most 64 million for one request by topic and declared base-model
identifier. Registration requires a positive supplied evaluation score, but public V1 will also
require cryptographically bound audit evidence before calling a pack "accepted." The frozen 4B
checkpoint remains unchanged and every pack is independently deletable. The current Ollama
launcher does not activate MLX packs. These are capacity ceilings, not quality claims.

Inspect and manage packs without loading a model:

```bash
.venv/bin/learnloop-experts list
.venv/bin/learnloop-experts route "Explain the Apollo computer" --base-model /exact/model/path --base-model-sha256 HASH
.venv/bin/learnloop-experts register training/candidate \
  --name apollo-history --topic apollo --topic history --evaluation evaluation.json
.venv/bin/learnloop-experts delete apollo-history
```

Registration counts the actual tensors in the capsule, refuses the 400M storage ceiling,
and requires a passing held-out evaluation containing frozen-suite and checkpoint SHA-256 IDs.

To chat through MLX with a routed parameter expert actually loaded, use:

```bash
.venv/bin/learnloop-mlx --model /exact/path/to/the/mlx-base-model
```

The command hashes the checkpoint, routes only compatible experts, unloads the previous
configuration before changing experts, and falls back to the frozen base weights when no
verified expert matches.

### Talk through Ollama

Qwen3-4B must be installed in Ollama. Then run:

```bash
uv sync --no-editable
.venv/bin/learnloop-ollama
```

### Locked 1,000-question quiz

The questions and private answer key are generated as separate files. Run the two
systems in separate processes so they are never in memory together:

```bash
.venv/bin/learnloop-make-quiz
.venv/bin/learnloop-run-quiz --mode base --limit 25
.venv/bin/learnloop-run-quiz --mode learnloop --limit 25
```

Runs are resumable. Each runner unloads Qwen from Ollama when it exits. Increase
`--limit` gradually; running all 1,000 questions will take many hours on a fanless laptop.

### Transparent 100-question comparison

The frozen suite contains 20 questions each from broad MMLU, ARC-Challenge, GSM8K,
college computer science, and generated exact-tool tasks. Public questions are fetched
through the official Hugging Face Dataset Viewer API and cached with source/license metadata.

```bash
.venv/bin/learnloop-benchmark-100 --build
.venv/bin/learnloop-benchmark-100 --model qwen3:4b-instruct-2507-q4_K_M --mode base
.venv/bin/learnloop-benchmark-100 --model qwen3:4b-instruct-2507-q4_K_M --mode learnloop
```

Measured on the development Mac, the fresh v2 suite scored 55/100 for base 4B and
71/100 for routed 4B plus LearnLoop. Verified tools improved 5/20 to 20/20 and official
retrieval improved MMLU 15/20 to 16/20, with no category regressions. General-skill
capsules were genuinely trained but failed to improve the fresh broad suite, so they were
rejected and deleted. This is not evidence of general 20B equivalence.

A separate official-web learning experiment retrieved licensed Apollo Guidance Computer
facts through Wikipedia's API, generated deterministic fact questions, and trained a
temporary 66 MB rank-8 update. With retrieval disabled, the untouched 4B answered 18/59
held-out phrasings correctly and the rewritten model answered 29/59: +11 questions,
+18.6 percentage points, or 1.61x the base task score. This demonstrates narrow temporary
fact retention in weights; it does not make the model 1.61x smarter in general.

The newer frozen v3 suite uses SHA-256
`6614b195138da8af925b59ae772d7f5499879b11b8197bca415a384e918c1274`.
On the same Ollama Qwen3-4B checkpoint, base scored 49/100 and the current routed framework
scored 70/100, with no category regressions: MMLU 12→12, ARC 17→17, GSM8K 7→12,
computer science 9→9, and exact tools 4→20. The framework uses more inference compute
on GSM8K and computer science; exact call/token budgets are recorded in v3's
`feature-evidence.json`.

An independently audited intermediate comparison against the installed Qwen2.5 7B
(7.6B dense parameters, Q4_K_M) scored 48/100 on those exact same 100 cases: MMLU 14,
ARC 18, GSM8K 4, computer science 12, and tools 0. This establishes a suite-specific
size-ladder checkpoint, not universal 7B superiority,
10B equivalence, 5x intelligence, or the required 20B result. The raw JSONL, summary,
manifest, and passing audit are kept in `benchmarks/benchmark-100-v3` and `configs`.

The strongest untouched evaluation is benchmark-100-v5, SHA-256
`49c6621e30016c4465edf50fd62f6e7b34b93c2ca72be8ffddf94ffc32d1a1c6`. Its 100 prompts
have zero overlap with both development suites. Base Qwen3-4B scored 52/100; locked
LearnLoop scored 75/100 (+23 points, 1.44x the base score) with no category regression.
On the same v5 prompts, Qwen2.5 7.6B scored 42/100, so LearnLoop achieved 1.79x that
reference's task score. All three raw runs pass independent audits. Temporary parameter
rewriting and official retrieval are separately verified capabilities; they were not invoked
to produce the 75/100 score, which came from arithmetic planning plus verified tools.

The integrated 159-case evaluation at `benchmarks/integrated-v1-7b` combines that untouched
100-case core with 59 held-out phrasings of facts learned from the official Wikipedia API.
Untouched 4B weights scored 70/159, LearnLoop scored 104/159, and Qwen2.5 7.6B scored
69/159. The candidate audit derives all three feature signals directly from raw rows:
official non-scraping retrieval, temporary parameter rewriting, and verified tools. Its strict
gate passes every identity, provenance, audit, feature, held-out-gain, score, and regression
check except one: 7.6B is below the required 18B minimum. This is therefore a real integrated
7B milestone, not a completed V1 or evidence of 20B/100x equivalence.

The v0.6 retrieval route crosses the 120/159 target without scraping. On the existing
integrated suite, deterministic matching against evidence fetched from Wikipedia's official
MediaWiki API scored 58/59 factual cases; combined with the frozen routed core's 75/100,
the audited score is **133/159**. A consistency run then generated 59 new factual cases from
a different topic through the same licensed API. Without case-specific tuning, it scored
54/59 and produced an independently audited **129/159** combined score (suite SHA-256
`90c4484ff3f50a59691a92e62800109197a804afde707bf09c7f27d5a9f057be`). The 100-case
core was reused, so this is a fresh factual extension rather than 159 entirely new prompts.
The gain is primarily retrieval and verified tools—not a claim that temporary weights alone
made the model 1.9x more intelligent.

### Consistent approximately-2x-size comparison

LearnLoop's 4.022B base plus framework was compared on identical questions with the installed
Qwen3 8.2B Q4_K_M checkpoint (2.039x as many parameters). It won three distinct evaluations:
73/100 versus 48/100 on v3, 75/100 versus 43/100 on v5, and 54/59 versus 19/59 on a fresh
official-web factual set. Recompute the machine gate with:

```bash
.venv/bin/learnloop-size-ladder-gate --reference-slug qwen3-8b \
  --reference-name "Qwen3 8.2B" --reference-parameters-b 8.2 \
  --output benchmarks/size-ladder-4b-vs-qwen3-8b.json
```

The two 100-question reference runs pass independent raw-row audits. This supports consistent
superiority to this particular Qwen3 8B checkpoint on these suites—not every 8B model,
universal equivalence, or "2x intelligence." The framework has tools and official retrieval;
the reference is the raw model under identical question and answer-format prompts.

DeepSeek-R1 8B is not included in this passing gate. Its Ollama template continued emitting
hidden reasoning until the output cap and produced no final answer in the fast evaluator, even
with thinking disabled. Those aborted rows are retained as rejected evidence rather than being
miscounted as wrong answers.

Run the full gated lifecycle with one command. It retrieves through the official API,
scores untouched weights, trains and scores temporary weights, accepts only a positive
gain, and deletes the source copy and capsule by default. Add `--keep-capsule` only when
an accepted update is still needed for inspection.

```bash
.venv/bin/learnloop-web-learning-cycle "history of the Apollo Guidance Computer" \
  --model /path/to/Qwen3-4B-Instruct-2507-4bit
```

The V1 release gate additionally requires an eligible 18B-or-larger reference, no category
regressions, active official retrieval, active temporary parameter rewriting, verified tools,
and a held-out gain caused by the weight rewrite. `current-v1-gate.json` intentionally fails
until those conditions are demonstrated.

The selected reference is `gpt-oss:20b`: 21B total parameters and 3.6B active parameters
per token (MoE), in Ollama's 14 GB MXFP4 artifact. Both parameter counts are disclosed.
Before downloading, run `.venv/bin/learnloop-reference-preflight`. It checks disk and runtime
memory separately. Downloading requires the 14 GB artifact plus a 5 GiB disk reserve. Local
execution requires at least 24 GiB physical memory: an observed 16 GiB M4 Air took 209 seconds
to load the weights and destabilized the system, so LearnLoop now refuses that unsafe run even
though Ollama lists 16 GiB as a minimum-compatible configuration.

After installation, `.venv/bin/learnloop-reference-run` runs `gpt-oss:20b` on the exact
v3 suite, resumes interrupted rows, unloads the model afterward, and independently audits
all 100 raw answers. The release gate requires the checked-in reference manifest plus
passing base, candidate, and reference audit reports; a manually typed parameter count is
not accepted.

On a 16 GiB machine, run `learnloop-reference-run` from the same checkout on a stronger Mac,
then copy its raw JSONL and attestation. Import and rescore those bytes locally without loading
the model:

```bash
.venv/bin/learnloop-reference-import \
  --suite benchmarks/benchmark-100-v3/suite.json \
  --results /path/to/results-reference-gpt-oss-20b-suite-6614b195138d.jsonl \
  --attestation /path/to/attestation-reference-gpt-oss-20b.json \
  --manifest configs/reference-gpt-oss-20b.json \
  --output-dir benchmarks/benchmark-100-v3
```

The importer verifies the model identity, frozen-suite hash, and exact raw-results hash, then
recomputes every score and runs the independent audit. The attestation is transparent
self-reported metadata, not proof of who operated the remote machine.

For the integrated V1 run, copy this checkout to a Mac with at least 24 GiB unified memory,
install/pull the manifest's model, and run one command:

```bash
.venv/bin/learnloop-reference-bundle \
  --create reference-gpt-oss-20b-bundle \
  --suite benchmarks/benchmark-100-v5/suite.json \
  --knowledge-cases benchmarks/web-weight-apollo/base.json \
  --reference-manifest configs/reference-gpt-oss-20b.json
```

Copy the resulting directory back and verify every byte before importing anything:

```bash
.venv/bin/learnloop-reference-bundle --verify reference-gpt-oss-20b-bundle
```

The bundle contains the exact frozen suite, 100 raw core answers, 59 raw closed-book
knowledge answers, model metadata, run attestation, and SHA-256 hashes. Bundle creation
refuses machines below 24 GiB; the 16 GiB Air cannot accidentally start the 20B model.

## Temporary weight rebuilding

The experimental rebuild pipeline performs real gradient training, fuses the learned
update into a standalone model, and deletes the temporary update. No adapter is needed
when running the rebuilt model:

```bash
.venv/bin/learnloop-rebuild             # safe dry run
.venv/bin/learnloop-rebuild --execute   # real training and fusion
```

Temporary training artifacts have a 5 GiB ceiling. Rebuilt candidates are never
promoted until they beat the base model on the locked test split.

After an incorrect answer, enter `/correct`. Ask a similar question again to test
whether the live memory changed the result.

## Live terminal test

```bash
uv sync --no-editable
.venv/bin/learnloop-live-test
```

The test prints the original answer, asks you for a correction, activates it while
the model remains loaded, and prints the new answer immediately underneath.

Run the repeatable base-model comparison with:

```bash
.venv/bin/learnloop-benchmark
```

Run the timed hard-reasoning stress test with:

```bash
.venv/bin/learnloop-stress-test --minutes 20
```

Export training data with:

```bash
uv run learnloop export
```

## Roadmap

1. Improve temporary rewrites on broad held-out skills without benchmark leakage
2. Expand deterministic fact extraction beyond four-digit years
3. Run the unchanged suite against a real approximately-20B quantized reference
4. Pass the strict V1 release gate

Before any public upload, run the local publication guard:

```bash
.venv/bin/learnloop-release-check
```

It refuses missing public metadata, root-level private/generated files, exposed local home
paths in public evidence, or a falsely passing V1 gate. A parent should still review the
final staged file list before publication.

## Safety design

Webpages and model-generated text are untrusted. Permanent training candidates must
have explicit approval and evidence. Every adapter will be versioned and evaluated
against a frozen test set before activation.
