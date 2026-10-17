# MatrixFractals: how it works, what worked, what did not

Written from the code, the git history (6 commits, 2026-02-19 to 2026-03-26) and the output left on disk
(about 430 saved matrices and 430 rendered videos). Statements marked *(inferred)* are conclusions from
that evidence, not recorded results.

## The idea

A fractal is defined by a small square matrix of cells. Each cell is one of:

- a complex constant, drawn uniformly from [-1.5, 1.5] on both axes;
- a function of a variable: `sin(C)`, `-cosh(Z)`, `inverse(D)`... with 23 functions available;
- a bare variable, optionally negated: `C`, `-D`, `Z`.

One iteration computes `Z <- sum over rows ( product over the row's cells )`. The variables are
`C` (the position in the complex plane), `D` (a third axis, time-like, `linspace(0,1,t) + i*sin(...)`)
and `Z` (the running value). A point "bails out" when `|Z|` leaves a threshold (4.0 in the renderer),
and the iteration at which that happens, plus the exit value, is the voxel. So the result is a 3D
volume, `x * y * t`, and every `D` slice is a Mandelbrot-style picture of the same formula.

## Pipeline

| Stage | File | Notes |
|---|---|---|
| Formula | `MatrixFormula.py` | Cells, random generation, crossover (split rows or columns), row shift, flip and targeted mutation. `process_matrix` is the iteration step. |
| Volume | `FractalTensors.py` | Batched CUDA/CPU iteration in windows of 45 M voxels. Resolution presets from `extreme_low_def` (12^3) to `render_desktop` (1024x1024x300). |
| Interestingness | `FractalTensors.evaluate_video_size` | Colormap the volume, encode as H.264 (crf 23), return bytes per voxel. High value = hard to compress = structure. |
| Search | `GeneticAlgorithm.py`, `main.simple_GA` | Pool, weighted pick (note^2), staged evaluation, truncation. |
| Render | `voxeltoVR.py` | Analytical ray marching straight from the formula (no stored volume), stereo side-by-side equirectangular, H.265, spatial-media metadata injection for VR players. |

### The staged evaluation (`GeneticAlgorithm.evaluate_matrix`)

1. Render at 32^3. Note = size ratio. If below 0.4, stop.
2. Render at 64^3. Add 2x the ratio. If below 0.2, stop.
3. Render at 256x256x64. Add 8x the ratio.

Most random matrices die at stage 1, so the expensive render only runs on formulas that already look structured.

### The GA loop (`main.simple_GA`)

Seed from the "kindle" files (good random matrices saved earlier by `generate_kindle`), then repeat:
measure pool similarity, adapt the flip rate, cross two weighted picks (2 children), occasionally shift rows,
truncate to the pool size. Every 1000 steps the top 5 are rendered at 1024x1024x300 and encoded.
When similarity exceeds 0.99 the process restarts itself with fresh randomness.

## What worked

- **Compression ratio as the fitness function.** It is cheap, needs no labels, and rejects both the
  empty case (everything bails out at once, so the video is flat) and the saturated case (nothing
  bails out). Roughly 430 matrices made it all the way to a rendered video *(inferred from `matrix_images`)*.
- **Staged, resolution-increasing evaluation.** It keeps nearly all the GPU time on promising formulas.
- **Kindle seeding.** Seeding the GA from pre-filtered random matrices (`note > 0.4`) skipped the long
  random start. The note in the file name records the score; the author ran 4 generators in parallel.
- **Windowed CUDA iteration.** The "fast cuda based inference" commit is what made 256x256x64 evaluations routine.
- **Analytical ray marching for output.** Marching rays through the formula itself removed the need for a
  stored 1024^3 volume and gave the VR videos (`vr*.mp4`, `output_360_sbs_tb_injected.mp4`).
- **Squared-note selection weight** gave a useful pressure toward the best without collapsing the pool at once.

## What did not work (or was never finished)

- **Random search** (`find_fractals_cuda`): replaced by the GA and left on a constructor signature that no longer
  exists. It found fractals, but slowly.
- **Targeted mutation** (`MatrixFormula.mutate_matrix_item`): `mutation_probability` is 0 in the config, and the code has
  a bug: `item.replace(...)` results are never assigned (strings are immutable), so function and variable switches
  and the polarity un-flip did nothing. Only complex scaling and "prepend a minus" had any effect. The GUI contains a
  corrected version.
- **Equirectangular projection modes of a stored volume**: the source comments record the outcome for each:
  `max_intensity`, `first_hit` and `depth_color` gave black frames, `isosurface` raised a broadcasting error.
  This is why rendering moved to analytical ray marching.
- **Point-cloud and "hires" projection**: written (`volume_to_equirectangular_pointcloud`, `voxel_to_vr_video_hires`)
  and called once on the precomputed `.bailout` files, but not part of the final path.
- **Non-square matrices**: `MatrixFormula((r, c))` indexes rows by `size[1]` but iterates `size[0]`, so only square
  matrices are consistent. All saved matrices are 4x4 or 5x5.
- **`drawFractal`**: uses `cm`, which is never imported; it fails when called.
- **`Projection.py`**: a 1D slope-sum projection prototype with no callers.
- **Pool convergence.** The code has an explicit escape hatch (restart when similarity > 0.99), which suggests
  the pool collapsed often *(inferred)*.

## Practical lessons

1. Define interestingness before you search. Compression ratio is a good first one.
2. Reject early at low resolution; structure that shows up at 32^3 usually survives at 256^3.
3. Keep a bank of pre-filtered seeds; a random start wastes the first hours.
4. Do not store volumes if a ray marcher can read the formula directly.
5. Check mutation operators with a unit test. A silent no-op looks like a hyperparameter that "does not matter".
6. Keep the score cheap and monotonic across resolutions so stage thresholds mean the same thing.

## Running it

- Original scripts: edit the `__main__` block of `main.py` (the choice is made by hostname) and run it with the
  repository's parent directory on `PYTHONPATH`, because modules import `src.X`.
- GUI: see the README. The GUI installs the import shims itself and works without `torchcodec` (it computes the
  same H.264 size ratio with `ffmpeg`).

## Hardware and dependencies

CUDA is strongly recommended; the ray marcher moves tensors to CUDA explicitly. Needs `torch`, `numpy`, `matplotlib`,
`imageio` (+ `imageio-ffmpeg`), `ffmpeg` on the path, and optionally `torchcodec` and `coolname`.

## Data formats

- `.frac`: a `torch.save` pickle of `{'fractal_name', 'matrix'}`. It unpickles `src.MatrixFormula`, so only open files you made.
- `.bailout`: a `torch.save` of the normalised bailout volume.
