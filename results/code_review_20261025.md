# Pre-training review — 2026-10-25

Scope: training CLI, PPO/GAE, model, environment, evaluation, Colab generator and embedded sources. Python syntax checked across 22 project files. This is one focused correctness review, not an exhaustive security or viewer audit. Production source was not edited; no training, GPU, package installation or download was performed.

## Findings

1. **P1 — Re-running the Colab launch cell starts concurrent training on the same output files.** `scripts/build_colab_notebook.py:161` calls `Popen` without checking whether the previous process is alive, then replaces `proc`. Both processes can write the same checkpoints and CSVs; the earlier process is no longer tracked by the notebook variable. Add an alive-process guard and output-directory ownership check before launching. Rebuild the notebook after fixing the generator.

2. **P1 — Resume can silently start a new model while appending to old metrics.** `mario_rl/ppo.py:140-150` loads only when `ckpt_latest.pt` exists, but decides append mode from the existence of `metrics.csv`. An interrupted run before the first checkpoint (saved every ten updates) leaves CSVs but no checkpoint. `resume=True` then starts at step zero with random weights and appends to the old history. Require a valid checkpoint before changing any output; validate compatible config before resume. `config.json` is currently overwritten before checkpoint validation at line 106.

3. **P2 — Evaluation silently returns fewer episodes and can bias comparisons.** `mario_rl/evaluation.py:121-131` applies `max_steps=20000` to the entire vector evaluation, then returns partial results without validating per-worker quotas. A fake environment completing one episode every 4000 steps reproduces **80 returned versus 100 requested**, with default 16 workers. Slow workers can be underrepresented. Enforce the requested quotas or explicitly report incomplete evaluation and avoid treating it as a completed benchmark.

4. **P2 — Allowed batch settings can produce NaNs or fail after environment launch.** `mario_rl/ppo.py:127,205-216` floors minibatch size and uses sample standard deviation. For example, `num_envs=1, rollout_len=9, minibatches=4` produces a final singleton batch; its normalized advantage is NaN. The review probe confirms the exact normalization expression is non-finite for one sample. More minibatches than samples yields a zero range step. Validate settings before allocating workers/buffers, and handle singleton batches or use a suitable population-standard-deviation calculation. Defaults do not trigger this issue.

5. **P2 — Resume forgets the best score and can replace a better checkpoint.** `mario_rl/ppo.py:157,264-266` resets `best_return` to negative infinity even when resuming. Once fifty new episodes finish, a worse model can overwrite `ckpt_best.pt`. Persist and restore the historical selection score, with compatible score semantics.

## Evidence and limits

- Existing suite: 5/5 passed, including GAE boundary/limit checks, PPO clipping gradients and one actual emulator episode with hand-calculated reward.
- Syntax: 22 Python files parsed successfully.
- Notebook: all 7 embedded source files match their local versions.
- Supplemental probes: `results/review_pretrain_20261025.py`; no real training or model checkpoint loading.
- Logs: `results/code-review-safety/20261025-220212-29848/stdout.log` (suite), `results/code-review-safety/20261025-220258-31528/stdout.log` (probes). Both guarded jobs finished with exit code 0 and zero active processes; peak job commit about 1.04 GiB and 1.02 GiB respectively, under a 1.5 GiB limit.
- Actual machine snapshot: approximately 49–52% physical RAM used, approximately 15–16 GiB available, over 11 GiB commit headroom. Destination D: had approximately 245 GiB free.
- Git reports that D:/Mario is not a repository, so no HEAD/diff baseline is available. No existing source files were changed.
- No end-to-end optimization smoke run, full training, GPU/Colab execution, or learning-quality validation. Passing unit tests does not certify training readiness. Installed runner documentation retains unresolved protections for heavy workloads; this review used only lightweight contained checks.

Recommendation: address findings 1–2 before uploading for a long run; fix 3 before relying on benchmark results, and cover 4–5 before changing batch settings or resuming experiments.
