# LearnLoop Reasoning Stress-Test Report

## Run

- Date: 2026-09-12
- Hardware: Apple Silicon, 16 GB unified memory
- Model: Qwen3-4B MLX 4-bit
- Timed final run: 15 minutes, following about 5 minutes of test-and-fix iterations
- Exact-answer comparisons completed: 9 pairs
- Average time per base/LearnLoop pair: 112.84 seconds

## Final results

| System | Correct | Accuracy | Missing final answer |
|---|---:|---:|---:|
| Base Qwen3-4B | 1/9 | 11.1% | 8/9 |
| LearnLoop | 2/9 | 22.2% | 7/9 |

The one-answer difference is not evidence of learned reasoning. No relevant reasoning
memory was attached to these generated puzzles, so generation variance is the likely cause.

## Errors found

1. Irrelevant US-history memory was originally attached to a grid puzzle because common
   words counted toward retrieval similarity.
2. Non-thinking mode produced quick but mostly incorrect guesses.
3. Thinking mode frequently exhausted 1,200 and then 2,200 tokens before returning a final answer.
4. Exact combinatorics and repeated modular arithmetic are inefficient when performed manually in tokens.
5. A factual memory layer does not create general reasoning improvement by itself.

## Improvements made during testing

1. Added stop-word filtering to prevent unrelated memory activation.
2. Added a regression test proving history memory does not attach to robot-grid questions.
3. Added a hard-task thinking switch to the model interface.
4. Changed the evaluation format to request `FINAL: number` before explanation.
5. Added missing-answer, memory-helped, memory-hurt, timing, and exact-correctness reporting.

## Next improvement

Add a sandboxed reasoning-tool loop. For arithmetic and counting tasks, the model should
write a small calculation, run it in a restricted environment, inspect the result, and
only then answer. This should be benchmarked against the frozen nine-question set before
any LoRA adapter is accepted.
