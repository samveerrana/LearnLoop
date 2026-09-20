# LearnLoop experiment card

Status: integrated 7.6B milestone passed; approximately-20B V1 gate pending external hardware.

## Research question

Can a local 4B model use a reversible combination of official retrieval, temporary parameter
updates, and verified tools to outperform its untouched checkpoint and a larger raw model on
a transparent mixed-task evaluation?

LearnLoop does **not** claim that a 4B checkpoint universally becomes a 7B, 20B, or 400B model.
Scores below apply only to the frozen suites and disclosed compute policy.

## System under test

- Base model: Qwen3-4B-Instruct-2507, 4-bit.
- Retrieval: official Wikipedia MediaWiki API with a CC-BY-SA-4.0 domain allowlist; no HTML scraping.
- Temporary rewrite: rank-8 update across all 36 layers, 16,515,072 trainable parameters
  (0.411% of the 4.022B checkpoint), 60 optimization steps, 66,115,678 bytes.
- Tools: restricted arithmetic-expression evaluator and deterministic exact solvers.
- Reversion: the retrieved source copy and temporary parameter capsule were deleted after
  evaluation; base weights remained unchanged.
- Safety: temporary artifacts have a 5 GiB ceiling. A measured 16 GiB Mac refuses the 20B
  reference after the first load destabilized the system; 24 GiB physical memory is required.

## Evaluation design

The integrated suite contains 159 cases:

- 100 frozen core cases: MMLU, ARC-Challenge, GSM8K, computer science, and exact-tool tasks.
- 59 closed-book fact-retention cases derived from allowlisted official-API text.
- Core v5 has zero prompt overlap with development suites v3 and v4.
- Fact-test wording is excluded from training, although the underlying facts are deliberately
  present in training; this measures retention, not discovery or broad reasoning.
- Retrieval is disabled while answering the 59 fact questions.
- Raw rows, expected answers, category totals, suite bytes, and model metadata are independently
  audited. Feature invocation is derived from raw traces rather than trusted checkboxes.

## Results

| System | Score | Percentage |
|---|---:|---:|
| Untouched 4B | 70/159 | 44.0% |
| LearnLoop 4B framework | 104/159 | 65.4% |
| Qwen2.5 7.6B Q4_K_M | 69/159 | 43.4% |

LearnLoop gained 34 correct answers over its untouched base (1.49x the base task score) and
35 over this 7.6B reference (1.51x its task score). There were no category regressions against
the base. On the isolated weight experiment, base scored 18/59 and temporary weights scored
29/59: +11 answers, +18.6 percentage points, or 1.61x the narrow task score.

## What caused the gain

- Verified arithmetic planning and exact tools produced most of the general-suite gain.
- Temporary weights produced the measured fact-retention gain.
- Official retrieval supplied licensed training evidence; it was not used to look up answers
  during closed-book evaluation.
- An experimental three-pass computer-science council was removed after a held-out regression.

## Strict gate status

The gate currently passes:

- reference identity and official provenance;
- exact frozen-suite identity and size;
- independent audits of all raw runs;
- candidate score at least as high as the current reference;
- no category regression versus base;
- raw-trace proof of retrieval, temporary rewriting, and tools;
- positive held-out gain from temporary weights.

It fails one requirement: the measured reference has 7.6B parameters, below the required 18B
minimum. Therefore V1 is not complete.

## Reproduction and next falsification test

Run the checked-in portable reference command on a Mac with at least 24 GiB memory:

```bash
.venv/bin/learnloop-reference-bundle \
  --create reference-gpt-oss-20b-bundle \
  --suite benchmarks/benchmark-100-v5/suite.json \
  --knowledge-cases benchmarks/web-weight-apollo/base.json \
  --reference-manifest configs/reference-gpt-oss-20b.json
```

Copy the bundle back, verify its SHA-256 manifest, import it, compose the 159-case reference,
and rerun the strict gate. A failure to match the 20B score is a real negative result, not a
reason to change the frozen suite.

## Relevance to an on-device team

The promising engineering idea is not “a small model magically becomes 100x larger.” It is a
resource-aware controller that chooses among unchanged inference, official retrieval, a small
temporary task update, and deterministic tools, then reverts costly state. A production study
should test broader tasks, latency, energy, privacy, catastrophic forgetting, adversarial source
text, update rollback, and multiple device classes before considering deployment.

## Evidence index

- Integrated gate: `benchmarks/integrated-v1-7b/v1-gate.json`
- Integrated raw rows and audits: `benchmarks/integrated-v1-7b/`
- Frozen zero-overlap core: `benchmarks/benchmark-100-v5/`
- Temporary-weight reports: `benchmarks/web-weight-apollo/`
- Reference manifests: `configs/reference-qwen2.5-7b.json`, `configs/reference-gpt-oss-20b.json`
