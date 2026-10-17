# Features that exist in the code but are not used by the current workflow

Each is reachable from the GUI, in the **Legacy / experimental** tab unless noted.

| # | Feature | Where in the code | Why unused | Status in the GUI |
|---|---|---|---|---|
| 1 | Random search | `main.find_fractals_cuda` | Superseded by the GA; the function calls `FractalTensors` without `conf` and no longer runs | Re-implemented (`bridge.random_search`), stage thresholds editable, saves the hit to `matrix_images/` and loads it into Explore |
| 2 | Hill climb | `main.test_laptop` | Early laptop-only experiment; same stale constructor | Re-implemented (`bridge.hill_climb`), live best matrix pushed to Explore |
| 3 | Targeted cell mutation | `MatrixFormula.mutate_matrix_item`, `main.simple_GA` | `mutation_probability = 0`; silent no-op bug | Checkbox in the GA tab, using the corrected `mutate_item_fixed` |
| 4 | Equirectangular volume projections | `voxeltoVR.volume_to_equirectangular` (modes `alpha_composite`, `first_hit`, `isosurface`, `depth_color`, `emission_absorption`, `max_intensity`) | Black frames or errors on the author's data | Callable on the volume rendered in Explore; failures are logged |
| 5 | Point-cloud projection | `voxeltoVR.volume_to_equirectangular_pointcloud`, `voxel_to_vr_video_hires` | Not part of the final path | "pointcloud" mode of the same panel |
| 6 | Per-layer PNG export | `FractalTensors.drawFractal` | Broken (`cm` undefined) | Replaced by "Export layers (PNG)" in Explore |
| 7 | Plain and streaming bailout videos | `FractalTensors.videoFractal`, `encode_video_streaming` | Replaced by the VR ray marcher; the first needs `torchcodec` | Not exposed: the Explore layer slider covers the use. The engine functions still work from code |
| 8 | Mid-size render presets | `Configuration` modes `test` / `middle` | Hard-coded to `render` | VR tab sets width, duration and fps directly |
| 9 | Row shift on non-square matrices | `MatrixFormula` size handling | Inconsistent; only square matrices work | The GUI only offers square matrices |
| 10 | 1D slope projection | `Projection.py` | Prototype, no callers | Not integrated: it has no working output (`somme_case` is incomplete) |
| 11 | Pre-computed bailout projection of `.bailout` files | `main.generate_multiple_projection_from_files` | One-off batch | Covered by item 4 on a live volume |
| 12 | Cubemap and `sbsto360.sh` stitching | `voxeltoVR.cubemap_*`, `sbsto360.sh` | Replaced by analytical equirectangular rendering | Not integrated |

Items 10 and 12 are left out on purpose: item 10 does nothing to wrap, item 12 is replaced by the active renderer.
