Plan 1 variants

This folder holds Plan 1 versions of the baselines. Each CUDA file is a thin wrapper that:
- defines PLAN_VARIANT=1
- includes the corresponding file from baselines/original/

You can branch behavior via conditional compilation inside the originals, e.g.:

#ifdef PLAN_VARIANT
// plan-specific tweaks
#endif
