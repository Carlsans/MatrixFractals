"""Glue between the PySide6 GUI and the original engine modules.

The engine imports itself both as ``src.X`` and as bare ``X`` and needs
``torchcodec`` and ``coolname``. This module makes those imports work from a
plain checkout without editing the engine, and holds the GUI-side versions of
workflows that only existed inside ``main.py`` (and used a stale API there).
"""
import random
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
KINDLE_DIR = REPO / "kindle"
IMAGES_DIR = REPO / "matrix_images"

DIMENSION_MODES = ["extreme_low_def", "very_low_def", "low_def", "medium_def",
                   "high_def", "higher_def", "render_laptop", "render_desktop"]
FUNCTIONS = ['sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'asin', 'acos', 'atan', 'asinh',
             'acosh', 'atanh', 'sinc', 'exp', 'square', 'sqrt', 'rsqrt', 'log', 'angle',
             'sigmoid', 'inverse', 'abs', 'relu']
VARIABLES = ['C', 'D', 'Z']
EQUIRECT_MODES = ["alpha_composite", "first_hit", "isosurface", "depth_color",
                  "emission_absorption", "max_intensity", "pointcloud"]
HAS_TORCHCODEC = False


def install_shims():
    """Make the engine importable. Safe to call more than once."""
    global HAS_TORCHCODEC
    if "src" not in sys.modules:
        pkg = types.ModuleType("src")
        pkg.__path__ = [str(REPO)]
        sys.modules["src"] = pkg
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

    try:
        import coolname  # noqa: F401
    except ImportError:
        stub = types.ModuleType("coolname")
        words = ["amber", "brave", "calm", "dusty", "eager", "fuzzy", "gentle", "hasty",
                 "icy", "jolly", "keen", "lucky", "mellow", "noble", "odd", "proud"]
        animals = ["fox", "owl", "seal", "toad", "wolf", "yak", "kiwi", "newt", "lynx", "crow"]
        stub.generate_slug = lambda: "-".join([random.choice(words), random.choice(words),
                                                random.choice(animals), str(random.randint(0, 9999))])
        sys.modules["coolname"] = stub

    try:
        import torchcodec.encoders  # noqa: F401
        HAS_TORCHCODEC = True
    except ImportError:
        pkg = types.ModuleType("torchcodec")
        enc = types.ModuleType("torchcodec.encoders")
        enc.VideoEncoder = object
        pkg.encoders = enc
        sys.modules["torchcodec"] = pkg
        sys.modules["torchcodec.encoders"] = enc

    # One class object for both import spellings, otherwise isinstance() and unpickling disagree.
    import src.MatrixFormula as mf
    import src.TimeFunctions as tf
    sys.modules["MatrixFormula"] = mf
    sys.modules["TimeFunctions"] = tf


install_shims()

from src.Configuration import Configuration  # noqa: E402
from src.FractalTensors import FractalTensors  # noqa: E402
from src.GeneticAlgorithm import GeneticAlgorithm  # noqa: E402
from src.MatrixFormula import MatrixFormula  # noqa: E402


def x264_size_ratio(self):
    """Same metric as FractalTensors.evaluate_video_size (x264 bytes per voxel, crf 23),
    computed with the ffmpeg binary when torchcodec is not installed."""
    frames = self.prepare_tensor_for_video().permute(0, 2, 3, 1).contiguous().numpy()  # (t, x, y, 3)
    t, h, w, _ = frames.shape
    cmd = ["ffmpeg", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", "10", "-i", "pipe:0", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
           "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p",
           "-movflags", "frag_keyframe+empty_moov", "-f", "mp4", "pipe:1"]
    out = subprocess.run(cmd, input=frames.tobytes(), capture_output=True, check=True).stdout
    return len(out) / (self.dimensions.x * self.dimensions.y * self.dimensions.t)


if not HAS_TORCHCODEC:
    FractalTensors.evaluate_video_size = x264_size_ratio


def default_conf():
    return Configuration()


def parse_cell(text):
    """Cell text -> engine item: complex tensor, 'fn(V)' or '[-]V'."""
    text = text.strip().replace(" ", "")
    if not text:
        raise ValueError("empty cell")
    body = text.lstrip("-")
    if "(" in body:
        fn, var = body[:-1].split("(")
        if fn not in FUNCTIONS or var not in VARIABLES:
            raise ValueError(f"unknown function or variable: {text}")
        return text
    if body in VARIABLES:
        return text
    value = complex(text.replace("i", "j"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.complex(torch.tensor([value.real], device=device, dtype=torch.float32),
                         torch.tensor([value.imag], device=device, dtype=torch.float32))


def cell_text(item):
    if torch.is_tensor(item):
        v = complex(item.reshape(-1)[0].item())
        return f"{v.real:.3f}{v.imag:+.3f}j"
    return str(item)


def random_matrix(conf, n):
    matrix = MatrixFormula(conf, (n, n))
    matrix.create_random_matrix()
    return matrix


def matrix_from_cells(conf, cells):
    n = len(cells)
    matrix = MatrixFormula(conf, (n, n))
    matrix.matrix = [[parse_cell(c) for c in row] for row in cells]
    return matrix


def load_frac(conf, path):
    """Load a .frac file. These are pickles: only open files you created yourself."""
    fractal = FractalTensors(conf, None, "loaded", (1, 1, 1), (0, 0), 2)
    fractal.load_fractal(fractal_name="", fullpath=str(path))
    return fractal.matrix


def save_frac(matrix, name, folder=IMAGES_DIR):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    torch.save({"fractal_name": name, "matrix": matrix}, folder / f"{name}.frac")
    return folder / f"{name}.frac"


def render(conf, matrix, dims, max_iter=100, name=None, with_score=True):
    """Compute a fractal volume. Returns (normalised bailout (x,y,t) float numpy, score or None, fractal)."""
    fractal = FractalTensors(conf, matrix, name or FractalTensors.generate_fractal_name(),
                             dims, (0, 0), 2, max_iter=max_iter)
    fractal.compute_fractal()
    bailout = fractal.bailout.float()
    span = (bailout.max() - bailout.min()).clamp(min=1e-9)
    volume = ((bailout - bailout.min()) / span).cpu().numpy()
    score = fractal.evaluate_video_size() if with_score else None
    return volume, score, fractal


def colorize(layer, cmap="gist_earth"):
    from matplotlib import colormaps
    rgba = colormaps.get_cmap(cmap)(np.nan_to_num(layer))
    return np.ascontiguousarray((rgba[..., :3] * 255).astype(np.uint8))


def mutate_item_fixed(matrix, rng=random):
    """Targeted mutation of one cell (the engine's mutate_matrix_item, minus its no-op bug).

    Engine bug: ``item.replace(...)`` results are discarded, so only the polarity flip that
    prepends '-' and the complex-scaling branch ever changed anything.
    Returns True when the caller should keep optimising the same cell (complex scaling).
    """
    i, j = rng.randrange(matrix.size[0]), rng.randrange(matrix.size[1])
    item = matrix.matrix[i][j]
    device = matrix.device
    if torch.is_tensor(item):
        factor = torch.complex(4 * torch.rand(1, device=device) - 2, 4 * torch.rand(1, device=device) - 2)
        matrix.matrix[i][j] = item * factor
        return True
    neg = item.startswith("-")
    body = item.lstrip("-")
    if "(" in body:
        fn, var = body[:-1].split("(")
        choice = rng.choice(["flip_polarity", "switch_function", "switch_variable"])
        if choice == "switch_function":
            fn = rng.choice(FUNCTIONS)
        elif choice == "switch_variable":
            var = rng.choice(VARIABLES)
        else:
            neg = not neg
        body = f"{fn}({var})"
    else:
        if rng.choice(["flip_polarity", "switch_variable"]) == "flip_polarity":
            neg = not neg
        else:
            body = rng.choice(VARIABLES)
    matrix.matrix[i][j] = ("-" if neg else "") + body
    return False


def clone_matrix(conf, matrix):
    clone = MatrixFormula(conf, tuple(matrix.size))
    clone.matrix = [[c.clone() if torch.is_tensor(c) else c for c in row] for row in matrix.matrix]
    return clone


def seed_pool(ga, conf, count, kindle_dir=KINDLE_DIR, min_note=0.4, matrix_size=5, job=None, log=print):
    """Fill the GA pool from kindle files when present, otherwise from random matrices
    that pass the stage-1 filter (what generate_kindle() does in main.py)."""
    files = sorted(Path(kindle_dir).glob("*.frac"), reverse=True) if Path(kindle_dir).is_dir() else []
    if files:
        log(f"seeding from {len(files)} kindle files")
        for path in random.choices(files, k=count):
            if job is not None and job.should_stop():
                return
            matrix = load_frac(conf, path)
            fractal, note = ga.evaluate_matrix(matrix)
            ga.add_item(fractal.get_fractal(), note)
        return
    log("no kindle files, generating random seeds")
    while len(ga.pool) < count:
        if job is not None and job.should_stop():
            return
        matrix = random_matrix(conf, matrix_size)
        fractal = FractalTensors(conf, matrix, FractalTensors.generate_fractal_name(),
                                 FractalTensors.get_dimensions_by_mode("low_def"), (0, 0), 2)
        fractal.compute_fractal()
        note = fractal.evaluate_video_size()
        if note > min_note:
            log(f"added seed {fractal.fractal_name} ({note:.3f})")
            ga.add_item(fractal.get_fractal(), note)


def ga_step(ga, conf, use_targeted_mutation=False, log=print):
    """One iteration of simple_GA() from main.py (similarity-driven flip rate, crossover,
    row shift, optional targeted mutation, truncation). The 'restart process' and automatic
    hi-def render of the original are left to the caller."""
    similarity = ga.sample_pool_similarity()
    if similarity <= .9:
        conf.mutation_flip_probability = 0.0001
    elif similarity < .95:
        conf.mutation_flip_probability = 0.01
    ga.apply_random_flip_mutation_entire_pool()

    if random.random() < conf.crossing_probability:
        first, second = ga.pick_item(), ga.pick_item()
        for child in MatrixFormula(conf, (1, 1)).cross_matrices(first['fractal']['matrix'],
                                                                second['fractal']['matrix']):
            fractal, note = ga.evaluate_matrix(child)
            ga.add_item(fractal.get_fractal(), note)

    picked = ga.pick_item()
    if random.random() < conf.row_shift_probability:
        picked['fractal']['matrix'].shift_rows()

    if use_targeted_mutation and random.random() < conf.mutation_probability:
        base = clone_matrix(conf, ga.pick_item()['fractal']['matrix'])
        for step in range(4):
            keep_going = mutate_item_fixed(base)
            fractal, note = ga.evaluate_matrix(base)
            ga.add_item({'fractal_name': FractalTensors.generate_fractal_name(),
                         'matrix': clone_matrix(conf, base)}, note)
            if not keep_going:
                break
    ga.kill_overpooled_items()
    log(f"pool {len(ga.pool)}  best {ga.pool[0]['note']:.3f}  similarity {similarity:.3f}")
    return similarity


def random_search(conf, thresholds=(.8, .7), job=None, log=print, size=4):
    """Legacy find_fractals_cuda(): random matrix, tiny test, mid test, then save + HD render.
    Thresholds are on the NormVar-normalised score exactly like the original."""
    from src.NormVar import NormVar
    norm1, norm2 = NormVar(), NormVar()
    norm1.set(.1)
    while job is None or not job.should_stop():
        matrix = random_matrix(conf, size)
        name = FractalTensors.generate_fractal_name()
        fractal = FractalTensors(conf, matrix, name, (24, 24, 24), (0, 0), 2)
        fractal.compute_fractal()
        norm1.set(fractal.evaluate_video_size())
        log(f"{name}: stage1 {norm1.get():.2f}")
        if norm1.get() <= thresholds[0]:
            continue
        fractal = FractalTensors(conf, matrix, name, (256, 256, 64), (0, 0), 2)
        fractal.compute_fractal()
        norm2.set(fractal.evaluate_video_size())
        log(f"{name}: stage2 {norm2.get():.2f}")
        if norm2.get() > thresholds[1]:
            path = save_frac(matrix, name)
            log(f"saved {path}")
            return path


def hill_climb(conf, size=4, dims=(24, 24, 24), job=None, log=print, on_best=None):
    """Legacy test_laptop(): mutate-and-keep-best on a single matrix, restarting from a small pool."""
    ga = GeneticAlgorithm(conf)
    best_matrix, highest, n = None, 0, 0
    while job is None or not job.should_stop():
        matrix = random_matrix(conf, size) if best_matrix is None else clone_matrix(conf, best_matrix)
        fractal = FractalTensors(conf, matrix, FractalTensors.generate_fractal_name(), dims, (0, 0), 2)
        fractal.compute_fractal()
        score = fractal.evaluate_video_size()
        if score > highest:
            highest, best_matrix = score, clone_matrix(conf, matrix)
            log(f"step {n}: new best {highest:.3f}")
            if on_best:
                on_best(best_matrix, highest)
        if n % 10 == 0 and best_matrix is not None:
            ga.add_item({'fractal_name': fractal.fractal_name, 'matrix': clone_matrix(conf, best_matrix)}, highest)
            ga.kill_overpooled_items()
            highest = 0
            best_matrix = clone_matrix(conf, ga.pick_item()['fractal']['matrix'])
        else:
            if best_matrix is not None:
                mutate_item_fixed(best_matrix)
        n += 1


class VRConfig:
    """Overrides applied to Configuration() inside voxeltoVR (out_W, fps, length)."""
    overrides = {}


def vr_modules():
    import src.voxeltoVR as vr
    if not getattr(vr, "_gui_patched", False):
        base = vr.Configuration

        class PatchedConfiguration(base):
            def __init__(self):
                super().__init__()
                self.__dict__.update(VRConfig.overrides)

        vr.Configuration = PatchedConfiguration
        vr._gui_patched = True
    return vr


def vr_preview(matrix, width=768, colormap="gist_earth", disparity=0.08):
    """One left-eye equirectangular frame of the ray-marched fractal (the active render path)."""
    vr = vr_modules()
    compiled = vr.compile_fractal_step_real(matrix)
    bounds = vr.find_fractal_bounds(compiled, currentDi=0.0, threshold=4.0, max_iter=50,
                                    search_range=3.0, resolution=64)
    if bounds is None:
        raise RuntimeError("no surface found: this formula has no visible fractal inside the search range")
    n_steps = 16
    return vr.analytical_ray_march_fast(
        compiled_step=compiled, currentDi=torch.tensor(0.0), x_difference=disparity, out_W=width,
        n_steps=n_steps, max_iter=50, threshold=4.0, step_size=bounds['radius'] / n_steps,
        col_chunk=102400, cmap_lut=vr.build_cmap_lut(colormap), accumulate=True)


def vr_export(matrix, path, duration, fps, width, disparity, colormap, max_iter=100):
    vr = vr_modules()
    VRConfig.overrides = {"out_W": width, "max_iter_render": max_iter}
    compiled = vr.compile_fractal_step_real(matrix)
    vr.voxel_to_vr_video(compiled, output_path=str(path), eye_disparity=disparity,
                         duration_sec=duration, fps=fps, colormap=colormap, equirectangular_mode=None)


def volume_projection(volume, mode, width=1024, colormap="gist_earth", threshold=0.15):
    """Legacy bailout-volume projections (volume (x,y,t) normalised numpy -> equirectangular frame)."""
    vr = vr_modules()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vol = torch.from_numpy(np.ascontiguousarray(volume)).float().to(device)
    lut = vr.build_cmap_lut(colormap)
    if mode == "pointcloud":
        return vr.volume_to_equirectangular_pointcloud(vol, lut, out_W=width, threshold=threshold)
    return vr.volume_to_equirectangular(vol, lut, out_W=width, mode=mode, threshold=threshold, col_chunk=64)
