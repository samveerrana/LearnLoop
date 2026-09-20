# LearnLoop public V1 checklist

Public V1 is useful only when every required item is reproducible. A benchmark score is
task-scoped and must never be described as a universal intelligence multiplier.

## Required before publishing

- [x] Frozen 4B checkpoint remains unchanged and recoverable.
- [x] Official, non-scraping retrieval is recorded with license metadata.
- [x] Temporary parameter rewriting has a positive held-out factual result (18/59 to 29/59).
- [ ] Expert admission recomputes scores and hashes from raw held-out rows instead of trusting supplied JSON.
- [x] Stored expert parameters are capped at 400M; active parameters are capped at 64M.
- [x] Oversized Ollama models are refused using a memory-aware 10 GiB runtime reserve.
- [x] Terminal launcher lists safe installed Ollama models for tools/retrieval/memory.
- [x] Installed-package launcher smoke test succeeds with one command after `uv sync --no-editable`.
- [x] Routed MLX expert activation is integrated into `learnloop-mlx`.
- [x] Raw benchmark rows, frozen suite hashes, audits, and failures are retained.
- [x] Candidate exceeds 120/159 twice: 133/159 existing and 129/159 with 59 freshly generated factual cases.
- [ ] Candidate matches or beats an eligible approximately-20B reference run externally.
- [x] A clean virtual-environment wheel installation test succeeds.
- [x] Clean-wheel installation succeeds in a separate environment with supported Python 3.11+.
- [x] Parent reviews repository for personal information, licenses, and account secrets.
- [x] User explicitly authorizes the GitHub push after the 2x-parameter gate and launcher pass.

## Claims allowed now

The current integrated result is 104/159 versus 70/159 for the untouched 4B model and
69/159 for the tested Qwen2.5 7.6B checkpoint. That is 1.49x the base task score on this
suite, not 1.49x general intelligence. The temporary rewrite separately improved one
narrow closed-book fact test from 18/59 to 29/59.

The 4.022B framework also beat the tested Qwen3 8.2B Q4_K_M checkpoint on three distinct
evaluations: 73/100 versus 48/100, 75/100 versus 43/100, and 54/59 versus 19/59. This is
an audited suite-specific 2x-parameter milestone, not a claim of 2x general intelligence.

Do not claim 2x intelligence, 10B/20B equivalence, or general superiority until suitable
untouched evaluations and an eligible external reference support those statements.
