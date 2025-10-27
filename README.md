Usage

- Original baselines (now under `baselines/original/`):
	- `make s1` -> builds and runs `baselines/original/baseline_1.cu`
	- `make s2` -> builds and runs `baselines/original/baseline_2.cu`
	- `make s3` -> builds and runs `baselines/original/baseline_3.cu`

- Plan variants:
	- Plan 1: `make s1_p1`, `make s2_p1`, `make s3_p1`
	- Plan 2: `make s1_p2`, `make s2_p2`, `make s3_p2`

Outputs go into `bin/`.

Notes

- Each plan's CUDA files are thin wrappers that define a `PLAN_VARIANT` macro and include the corresponding file from `baselines/original/`. You can branch behavior inside the originals with `#ifdef PLAN_VARIANT`.
- You can override the GPU architecture when building, e.g. `make ARCH='-arch=sm_90' s1`.