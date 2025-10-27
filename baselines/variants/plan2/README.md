Plan 2 variants

This folder holds Plan 2 versions of the baselines. Each CUDA file is a thin wrapper that:
- defines PLAN_VARIANT=2
- includes the corresponding file from baselines/original/

Use #ifdef PLAN_VARIANT guards in the originals to switch behaviors for this plan.
