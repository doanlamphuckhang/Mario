# Pre-training fixes — 2026-10-25

Implemented all five findings from code_review_20261025.md:

- Notebook refuses a second launch while its process is alive. An OS-backed exclusive lock protects the run directory even for separate CLI launches. Fresh training refuses a nonempty run directory.
- Resume requires a valid checkpoint and compatible configuration before rewriting config/logs or launching environments. Logs ahead of the checkpoint are preserved in timestamped backups before trimming. Missing log headers are recreated independently.
- Vector evaluation now enforces episode quotas with a per-episode step limit. Incomplete evaluation raises an explicit error and closes workers.
- Batch divisibility and minimum minibatch size are checked before creating outputs/workers, preventing singleton normalization and zero-step minibatch iteration. Positive integer settings and minimum rollout budget are checked.
- Checkpoints persist best score and recent episode history. Resume also checks the best checkpoint, including legacy score recovery from matching CSV rows, preventing a worse resumed model from replacing the historical best.

Related reliability fixes: checkpoint writes use temporary files and atomic replacement; training resources close on initialization/reset failures after construction; notebook monitoring tolerates absent/empty initial metrics and reports a nonzero training exit. The regenerated notebook includes the new run_io module (8 embedded files).

## Verification

- Nine new regression scenarios passed across separate runs. Eight passed in results/fix-safety/20261025-220852-31564/stdout.log; the remaining synthetic training/resume test passed after fixing its test fixture in results/fix-safety/20261025-221011-32304/stdout.log. That test uses a two-parameter CPU model and fake environments, with actual optimizer/checkpoint handling.
- The original five tests passed again: results/fix-safety/20261025-221027-31012/stdout.log.
- New coverage includes cross-process lock refusal/release, no-op protection on missing/incompatible resume, backup preservation, legacy best score recovery, quota completion, explicit timeout cleanup, notebook rerun prevention and embedded-source equality.
- One full-suite attempt was stopped by the installed runner for system commit headroom. Its logs and status are preserved at results/fix-safety/20261025-220929-26228/. It ended with zero active processes. After checking memory, only the missing lightweight test was rerun with a 1536 MiB cap (peak job commit 1132 MiB); no full-suite retry or real training followed.
- Existing files were snapshotted before editing in results/pretrain-fix-backup-20261025/. D:/Mario has no Git repository; no commit baseline exists.

No GPU, downloads, package changes, real model training or Colab execution. Windows locking was exercised; the Linux flock branch awaits execution on Colab. Resume does not restore emulator/RNG state exactly. Checkpoint-based continuation may repeat work since the last checkpoint; original excess logs remain in backups.
