"""
voxel_to_vr.py
==============
Convert a normalized (D, H, W) voxel tensor to a side-by-side stereo VR video.

Pipeline
--------
  1. Mean Intensity Projection along D  →  (H, W)  flat projection
  2. Depth map via value-weighted D index  →  (H, W) in [0, 1]
  3. Apply viridis / plasma colormap  →  (H, W, 3) uint8
  4. Warp left & right eyes with grid_sample  (parallax from depth map)
  5. Concatenate side-by-side and write .mp4

Install dependencies
--------------------
  pip install torch numpy matplotlib imageio imageio-ffmpeg

Usage
-----
  python voxel_to_vr.py              # runs the built-in demo with random data
  # or import and call voxel_to_vr_video() directly with your own tensor
"""
import os
import shutil
import struct

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import imageio
from pathlib import Path

from src.Configuration import Configuration


# ──────────────────────────────────────────────────────────────────────────────
# Core projection helpers
# ──────────────────────────────────────────────────────────────────────────────

def mean_projection(volume: torch.Tensor) -> torch.Tensor:
    """
    Mean Intensity Projection along the depth axis.
    (D, H, W) → (H, W), values in [0, 1]
    """
    return volume.mean(dim=0)


def depth_map(volume: torch.Tensor) -> torch.Tensor:
    """
    Value-weighted mean of the normalised depth index.

    Each voxel's weight is its scalar value; the depth index runs from
    0.0 (front slice, D=0) to 1.0 (back slice, D=D-1).

    Low depth  → close to the camera  → larger stereo disparity
    High depth → far from the camera  → smaller stereo disparity

    Returns (H, W) in [0, 1].
    """
    D = volume.shape[0]
    idx = torch.linspace(0, 1, D, device=volume.device).view(D, 1, 1)
    weights = volume.clamp(min=0)
    denom = weights.sum(dim=0).clamp(min=1e-8)
    return (weights * idx).sum(dim=0) / denom  # (H, W)


def apply_colormap(gray: torch.Tensor, cmap_name: str = "plasma") -> np.ndarray:
    """
    Apply a matplotlib colormap to a (H, W) float tensor in [0, 1].
    Returns (H, W, 3) uint8.
    """
    cmap = plt.get_cmap(cmap_name)
    rgb = cmap(gray.cpu().numpy())[:, :, :3]  # drop alpha channel
    return (rgb * 255).astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# Stereo warp  (uses grid_sample for differentiable, sub-pixel accuracy)
# ──────────────────────────────────────────────────────────────────────────────

def warp_eye(
        image_np: np.ndarray,  # (H, W, 3) uint8
        d_map: torch.Tensor,  # (H, W) float [0, 1]
        disparity: float,  # signed; + = left eye, - = right eye
) -> np.ndarray:
    """
    Shift every pixel horizontally by  d_map * disparity  using grid_sample.

    Depth-based parallax:
      Voxels concentrated at the FRONT (d_map near 0) → small shift
      Voxels concentrated at the BACK  (d_map near 1) → large shift

    The shift is expressed in normalised image coordinates [-1, 1], so
    disparity=0.04 equals ~2% of the image width at maximum depth.
    """
    H, W = d_map.shape
    device = d_map.device

    # (H, W, 3) uint8 → (1, 3, H, W) float32
    img_t = (
            torch.from_numpy(image_np).float()
            .permute(2, 0, 1).unsqueeze(0) / 255.0
    ).to(device)

    # Identity sampling grid in normalised coords [-1, 1]
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)

    # Shift x proportionally to depth
    new_x = (grid_x + d_map * disparity).clamp(-1, 1)
    grid = torch.stack([new_x, grid_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)

    warped = F.grid_sample(
        img_t, grid,
        mode="bilinear",
        padding_mode="border",  # replicate edge pixels — no black borders
        align_corners=True,
    )  # → (1, 3, H, W)

    return (warped.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

def depth_colored_projection(volume: torch.Tensor) -> np.ndarray:
    """
    Replace the two-step mean_projection + apply_colormap with a single
    HSV composite that encodes depth into hue.

    H (hue)        = value-weighted mean depth  [0, 1]  → full color wheel
                     0.0 = red (near)  |  0.33 = green  |  0.66 = blue (far)
    S (saturation) = fixed at 1.0
    V (value)      = mean intensity projection  [0, 1]

    Returns (H, W, 3) uint8.
    """
    # intensity: mean across D  →  (H, W)
    intensity = volume.mean(dim=0)

    # hue: value-weighted depth index  →  (H, W)
    D = volume.shape[0]
    idx = torch.linspace(0, 1, D, device=volume.device).view(D, 1, 1)
    weights = volume.clamp(min=0)
    denom = weights.sum(dim=0).clamp(min=1e-8)
    hue = (weights * idx).sum(dim=0) / denom  # (H, W) in [0, 1]

    H_img, W_img = intensity.shape
    h = hue.cpu().numpy()
    v = intensity.cpu().numpy()

    # Vectorised HSV → RGB  (S=1 everywhere)
    # Using matplotlib's hsv colormap for H, then modulate V
    hsv_img = np.stack([h, np.ones_like(h), v], axis=-1)   # (H, W, 3)
    rgb = plt.cm.hsv(h)[:, :, :3] * v[:, :, None]          # hue * brightness
    return (rgb.clip(0, 1) * 255).astype(np.uint8)

def build_cmap_lut(cmap_name: str, n: int = 256) -> torch.Tensor:
    """
    Pre-bake a matplotlib colormap into a (n, 3) float32 GPU tensor.
    Call once at startup, reuse every frame.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cmap = plt.get_cmap(cmap_name)
    lut  = cmap(np.linspace(0, 1, n))[..., :3]          # (n, 3) on CPU
    return torch.from_numpy(lut).float().to(device)      # → GPU

def voxel_colored_projection(
    volume: torch.Tensor,
    cmap_lut: torch.Tensor,          # (256, 3) float32 on CUDA  ← pre-baked
) -> np.ndarray:
    """
    Same front-to-back alpha compositing, but colormap lookup
    stays entirely on GPU via integer LUT indexing.
    """
    device = volume.device

    n = cmap_lut.shape[0]
    D = volume.shape[0]

    out = torch.zeros(volume.shape[1], volume.shape[2], 3, device=device)  # (H, W, 3)
    T = torch.ones(volume.shape[1], volume.shape[2], device=device)  # transmittance

    for d in range(D):
        alpha_d = volume[d].clamp(0, 1)  # (H, W)
        idx_d = (alpha_d * (n - 1)).long()  # (H, W)
        color_d = cmap_lut[idx_d]  # (H, W, 3)
        w = (T * alpha_d).unsqueeze(-1)  # (H, W, 1)
        out += w * color_d
        T *= (1.0 - alpha_d)

    return (out.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

def make_rotation_matrix_y(angle_deg: float, device) -> torch.Tensor:
    angle = torch.tensor(angle_deg * np.pi / 180.0)
    c, s  = torch.cos(angle).item(), torch.sin(angle).item()
    return torch.tensor([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c],
    ], dtype=torch.float32, device=device)

def rotate_volume_y(volume: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """
    Rotate a (D, H, W) volume around its Y axis (the H axis) using grid_sample.
    angle_deg : rotation angle in degrees
    Returns a (D, H, W) tensor — same shape, rotated.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D, H, W = volume.shape
    angle = torch.tensor(angle_deg * np.pi / 180.0,device=device)
    c, s = torch.cos(angle), torch.sin(angle)

    # Affine matrix for Y-axis rotation (acts on the D-W plane)
    # grid_sample expects (N, C, D, H, W) for 5D, so we treat D and W as the
    # rotation plane and H as the neutral axis.
    theta = torch.tensor([
        [ c,  0,  s, 0],
        [ 0,  1,  0, 0],
        [-s,  0,  c, 0],
    ], dtype=torch.float32,device=device).unsqueeze(0)  # (1, 3, 4)

    # volume → (1, 1, D, H, W)
    vol5 = volume.unsqueeze(0).unsqueeze(0)

    grid = F.affine_grid(theta, vol5.shape, align_corners=True)  # (1, D, H, W, 3)
    rotated = F.grid_sample(
        vol5, grid,
        mode="bilinear",
        padding_mode="zeros",   # outside the volume → 0 (empty space)
        align_corners=True,
    )
    return rotated.squeeze(0).squeeze(0)  # (D, H, W)
def project_half(half_volume: torch.Tensor, cmap_lut: torch.Tensor) -> torch.Tensor:
    """
    Front-to-back composite of a half-volume (center → face).
    half_volume : (D/2, H, W) — already sliced and ordered center-first.
    Returns (H, W, 3) float32 on device.
    """
    device = half_volume.device
    n      = cmap_lut.shape[0]
    out    = torch.zeros(*half_volume.shape[1:], 3, device=device)
    T      = torch.ones (*half_volume.shape[1:],    device=device)

    for d in range(half_volume.shape[0]):
        alpha = half_volume[d].clamp(0, 1)
        color = cmap_lut[(alpha * (n - 1)).long()]
        w     = (T * alpha).unsqueeze(-1)
        out  += w * color
        T    *= (1.0 - alpha)

    return out.clamp(0, 1)
def cubemap_from_center(volume: torch.Tensor, cmap_lut: torch.Tensor) -> dict:
    """
    Project outward from the center of the volume along all 3 axes.
    Returns 6 faces as (H, W, 3) uint8 numpy arrays.

    Axis convention:
      D axis (+front / -back)  →  volume[d_mid:] / volume[:d_mid].flip(0)
      H axis (+down  / -up  )  →  volume[:,h_mid:,:] etc.
      W axis (+right / -left)  →  volume[:,:,w_mid:] etc.
    """
    D, H, W = volume.shape
    dc, hc, wc = D // 2, H // 2, W // 2

    def proj(half):
        return (project_half(half, cmap_lut).cpu().numpy() * 255).astype(np.uint8)

    return {
        "+D": proj(volume[dc:  , :, :]),           # look forward
        "-D": proj(volume[:dc  , :, :].flip(0)),   # look backward
        "+H": proj(volume[:, hc:  , :].permute(1, 0, 2)),   # look down
        "-H": proj(volume[:, :hc  , :].flip(1).permute(1, 0, 2)),  # look up
        "+W": proj(volume[:, :, wc:  ].permute(2, 0, 1)),   # look right
        "-W": proj(volume[:, :, :wc  ].flip(2).permute(2, 0, 1)),  # look left
    }
def stitch_cubemap(faces: dict, face_H: int, face_W: int) -> np.ndarray:
    """
    Arrange faces in a cross layout:

            [ -H ]
      [-W]  [ -D ]  [+W]  [+D]
            [ +H ]

    Returns (3*face_H, 4*face_W, 3) uint8.
    """
    empty = np.zeros((face_H, face_W, 3), dtype=np.uint8)
    row0  = np.concatenate([empty,        faces["-H"], empty,        empty       ], axis=1)
    row1  = np.concatenate([faces["-W"],  faces["-D"], faces["+W"],  faces["+D"] ], axis=1)
    row2  = np.concatenate([empty,        faces["+H"], empty,        empty       ], axis=1)
    return np.concatenate([row0, row1, row2], axis=0)
def cubemap_to_equirectangular(faces: dict, out_W: int = 2048) -> np.ndarray:
    """
    Remap 6 cubemap faces to an equirectangular (spherical) projection.
    out_W : output width in pixels  (out_H = out_W // 2)

    For every output pixel, compute its (azimuth, elevation) angle,
    convert to a 3D ray direction, find the dominant face, sample it.
    Returns (out_H, out_W, 3) uint8.
    """
    out_H = out_W // 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pre-stack all faces into a single tensor  (6, H, W, 3) → (6, 3, H, W)
    order   = ["+D", "-D", "+W", "-W", "+H", "-H"]
    face_np = np.stack([faces[k] for k in order], axis=0)           # (6, H, W, 3)
    faces_t = torch.from_numpy(face_np).float().permute(0, 3, 1, 2) # (6, 3, H, W)
    faces_t = faces_t.to(device) / 255.0

    # Spherical coordinates for every output pixel
    az  = torch.linspace(-np.pi,    np.pi,  out_W, device=device)   # [-π, π]
    el  = torch.linspace( np.pi/2, -np.pi/2, out_H, device=device)  # [π/2, -π/2]
    el_g, az_g = torch.meshgrid(el, az, indexing="ij")               # (H, W)

    # Ray direction in 3D  (x=right, y=up, z=forward)
    x =  torch.cos(el_g) * torch.sin(az_g)
    y =  torch.sin(el_g)
    z =  torch.cos(el_g) * torch.cos(az_g)

    # Find dominant face and 2D UV per pixel
    ax, ay, az_ = x.abs(), y.abs(), z.abs()
    dominant = torch.stack([az_, az_, ax, ax, ay, ay], dim=0).argmax(dim=0)  # (H,W)

    # UV lookup per face  (u, v in [-1, 1])
    uv_map = {
        0: ( x / z.abs(),  y / z.abs()),  # +D  (z > 0)
        1: (-x / z.abs(),  y / z.abs()),  # -D  (z < 0)
        2: (-z / x.abs(),  y / x.abs()),  # +W  (x > 0)
        3: ( z / x.abs(),  y / x.abs()),  # -W  (x < 0)
        4: ( x / y.abs(), -z / y.abs()),  # +H  (y > 0)
        5: ( x / y.abs(),  z / y.abs()),  # -H  (y < 0)
    }

    out_rgb = torch.zeros(out_H, out_W, 3, device=device)

    for face_idx in range(6):
        mask      = (dominant == face_idx)                           # (H, W)
        if not mask.any():
            continue
        u, v      = uv_map[face_idx]                                 # (H, W) each
        grid      = torch.stack([u, v], dim=-1).unsqueeze(0)        # (1, H, W, 2)

        # Sample this face for ALL pixels, then mask in only the relevant ones
        sampled   = F.grid_sample(
            faces_t[face_idx].unsqueeze(0),                          # (1, 3, fH, fW)
            grid.clamp(-1, 1),
            mode="bilinear", padding_mode="border", align_corners=True
        ).squeeze(0).permute(1, 2, 0)                                # (H, W, 3)

        out_rgb[mask] = sampled[mask]

    return (out_rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
def inject_spatial_metadata(input_path: str, output_path: str, stereo: bool = True):
    """
    Inject XMP spatial metadata using ffmpeg so that Facebook, YouTube,
    and Meta Quest recognise the file as a 360° video.

    Requires ffmpeg in PATH.
    stereo=True  → side-by-side left/right equirectangular (for VR headsets)
    stereo=False → mono equirectangular (for flat 360° browsing)
    """
    import subprocess
    stereo_meta = (
        "stereo_mode=left-right:"
        "spherical=true:"
        "projection=equirectangular"
        if stereo else
        "spherical=true:projection=equirectangular"
    )
    subprocess.run([
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", "v360=input=e:output=e",          # passthrough, forces metadata
        "-metadata:s:v", stereo_meta,
        "-codec", "copy",
        output_path,
    ], check=True)

def inject_spatial_metadata(input_path: str, output_path: str, stereo: bool = True):
    import subprocess
    input_path = Path(input_path)
    output_path = Path(output_path)
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)
    spatial_media_dir = "/home/carl/PycharmProjects/MatrixFractals/src/spatial-media"

    stereo_flag = "--stereo=left-right" if stereo else "--stereo=none"

    result = subprocess.run(
        [
            "python", "-m", "spatialmedia",
            "-i", stereo_flag,
            input_path,
            output_path,
        ],
        cwd=spatial_media_dir,  # must run from the repo root
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("[spatial-media stderr]", result.stderr)
        print("[spatial-media stdout]", result.stdout)
        raise RuntimeError("spatial-media injection failed")

    print(f"[inject_metadata] saved → {output_path}")

    # Verify it worked
    verify = subprocess.run(
        ["python", "-m", "spatialmedia", output_path],
        cwd=spatial_media_dir,
        capture_output=True, text=True,
    )
    print("[verify]", verify.stdout.strip())
def volume_to_equirectangular_pointcloud(
    volume: torch.Tensor,
    cmap_lut: torch.Tensor,
    out_W: int = 2048,
    threshold: float = 0.02,
    base_radius: float = 3,
    max_radius: int = 12,
    gamma: float = 1.0,
    max_points: int = 300000,
) -> np.ndarray:
    device = volume.device
    out_H  = out_W // 2
    D, H, W = volume.shape
    n = cmap_lut.shape[0]

    # ── 1. Extract on CPU to avoid GPU OOM ───────────────────────────────────
    vol_cpu = volume.cpu()
    mask    = vol_cpu >= threshold
    d_idx, h_idx, w_idx = mask.nonzero(as_tuple=True)  # all on CPU
    vals    = vol_cpu[d_idx, h_idx, w_idx]

    if len(vals) == 0:
        print("[pointcloud] no voxels above threshold — lower it")
        return np.zeros((out_H, out_W, 3), dtype=np.uint8)

    print(f"[pointcloud] active voxels: {len(vals):,}  threshold={threshold}")

    # Subsample on CPU — cheap, no GPU memory needed
    if len(vals) > max_points:
        step  = max(1, len(vals) // max_points)        # deterministic stride
        d_idx, h_idx, w_idx = d_idx[::step], h_idx[::step], w_idx[::step]
        vals  = vals[::step]
        print(f"[pointcloud] subsampled to: {len(vals):,}")

    # Move only the subsampled points to GPU
    d_idx = d_idx.to(device)
    h_idx = h_idx.to(device)
    w_idx = w_idx.to(device)
    vals  = vals.to(device)

    # ── 2. Voxel coords → 3D centred space ───────────────────────────────────
    x =  (w_idx.float() / (W - 1)) * 2 - 1
    y = -((h_idx.float() / (H - 1)) * 2 - 1)
    z =  (d_idx.float() / (D - 1)) * 2 - 1
    dist = (x**2 + y**2 + z**2).sqrt().clamp(min=1e-6)

    # ── 3. Sort far → near ────────────────────────────────────────────────────
    order = dist.argsort(descending=True)
    x, y, z = x[order], y[order], z[order]
    dist, vals = dist[order], vals[order]

    # ── 4. Equirectangular projection ─────────────────────────────────────────
    az = torch.atan2(x, z)
    el = torch.asin((y / dist).clamp(-1, 1))
    px = ((az / (2 * torch.pi) + 0.5) * out_W).long().clamp(0, out_W - 1)
    py = ((0.5 - el / torch.pi) * out_H).long().clamp(0, out_H - 1)

    # ── 5. Splat radius per point ─────────────────────────────────────────────
    radius = (base_radius / dist).clamp(1, max_radius)    # float, (N,)

    # ── 6. Colors ─────────────────────────────────────────────────────────────
    cidx   = (vals.clamp(0, 1) ** gamma * (n - 1)).long()
    colors = cmap_lut[cidx]                               # (N, 3)

    # ── 7. Vectorised disk splat ──────────────────────────────────────────────
    # Build offsets for the largest possible disk once
    r_int  = int(max_radius)
    offsets = [(dy, dx)
               for dy in range(-r_int, r_int + 1)
               for dx in range(-r_int, r_int + 1)
               if dx*dx + dy*dy <= r_int * r_int]

    canvas = torch.zeros(out_H * out_W, 3, device=device)

    for dy, dx in offsets:
        d2       = dx*dx + dy*dy
        # Only points whose radius reaches this offset
        in_splat = (radius * radius) >= d2
        if not in_splat.any():
            continue
        ty   = (py[in_splat] + dy).clamp(0, out_H - 1)
        tx   = (px[in_splat] + dx).clamp(0, out_W - 1)
        flat = ty * out_W + tx
        canvas.index_put_((flat,), colors[in_splat], accumulate=False)

    return (canvas.view(out_H, out_W, 3).clamp(0, 1).cpu().numpy() * 255
            ).astype(np.uint8)

def volume_to_equirectangular(
            volume: torch.Tensor,
            cmap_lut: torch.Tensor,
            out_W: int = 2048,
            n_steps: int = 256,
            gamma: float = 1.0,
            col_chunk: int = 128,
            mode: str = "first_hit",  # see modes below
            threshold: float = 0.15,  # used by first_hit, isosurface
            opacity_scale: float = 8.0,  # used by emission_absorption — higher = more opaque
    ) -> np.ndarray:
        """
        Spherical raycasting with multiple rendering modes:

          "max_intensity"        — brightest voxel along each ray (MIP).
                                   Sharpest, reveals all structure, ignores depth. ## Black screen for all frames

          "first_hit"            — first voxel above `threshold` (surface rendering).
                                   Crisp surfaces, good for dense fractal shells.## Black screen for all frames

          "isosurface"           — like first_hit but shades using the local gradient
                                   as a surface normal, giving a 3D lit appearance. ## Broadcasting error
                                   [voxel_to_vr] frame size : 1024x512  (side-by-side)


          "depth_color"          — color = depth of the max-value voxel, brightness = value.
                                   Reveals spatial layout without blurring. ## Black screen for all frames

          "emission_absorption"  — physically based: each voxel emits and absorbs light.
                                   `opacity_scale` controls how quickly the ray is absorbed.
                                   More structured than alpha_composite. ## Black screen for all frames

          "alpha_composite"      — original front-to-back alpha composite (softest/blurriest).
        """
        device = volume.device
        out_H = out_W // 2
        n = cmap_lut.shape[0]

        az_all = torch.linspace(-torch.pi, torch.pi, out_W, device=device)
        el_all = torch.linspace(torch.pi / 2, -torch.pi / 2, out_H, device=device)
        t = torch.linspace(0.0, 1.0, n_steps, device=device)

        out_np = np.zeros((out_H, out_W, 3), dtype=np.uint8)

        # Pre-compute gradient volume for isosurface shading (once, outside loop)
        if mode == "isosurface":
            # Central differences along each axis → (3, D, H, W)
            gx = torch.zeros_like(volume)
            gy = torch.zeros_like(volume)
            gz = torch.zeros_like(volume)
            gx[:, :, 1:-1] = (volume[:, :, 2:] - volume[:, :, :-2]) * 0.5
            gy[:, 1:-1, :] = (volume[:, 2:, :] - volume[:, :-2, :]) * 0.5
            gz[1:-1, :, :] = (volume[2:, :, :] - volume[:-2, :, :]) * 0.5
            grad_vol = torch.stack([gx, gy, gz], dim=0)  # (3, D, H, W)

        for col_start in range(0, out_W, col_chunk):
            col_end = min(col_start + col_chunk, out_W)
            cW = col_end - col_start
            az = az_all[col_start:col_end]

            el_g, az_g = torch.meshgrid(el_all, az, indexing="ij")  # (out_H, cW)

            rx = torch.cos(el_g) * torch.sin(az_g)
            ry = torch.sin(el_g)
            rz = torch.cos(el_g) * torch.cos(az_g)

            sx = rx.unsqueeze(0) * t[:, None, None]  # (n_steps, out_H, cW)
            sy = ry.unsqueeze(0) * t[:, None, None]
            sz = rz.unsqueeze(0) * t[:, None, None]

            grid = torch.stack([sx, sy, sz], dim=-1).unsqueeze(0)  # (1, n_steps, out_H, cW, 3)
            vol5 = volume.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)

            sampled = F.grid_sample(
                vol5, grid,
                mode="bilinear", padding_mode="zeros", align_corners=True,
            ).squeeze(0).squeeze(0)  # (n_steps, out_H, cW)

            # ── rendering modes ────────────────────────────────────────────────

            if mode == "max_intensity":
                vals, _ = sampled.max(dim=0)
                vals = vals.clamp(0, 1) ** gamma
                idx = (vals * (n - 1)).long()
                out_rgb = (cmap_lut[idx] * 255).clamp(0, 255).byte()  # ← * 255

            elif mode == "first_hit":
                hit_mask = sampled >= threshold
                hit_idx = hit_mask.long().argmax(dim=0)
                hit_any = hit_mask.any(dim=0)
                vals = sampled.gather(0, hit_idx.unsqueeze(0)).squeeze(0)
                vals = (vals.clamp(0, 1) ** gamma) * hit_any.float()
                idx = (vals * (n - 1)).long()
                out_rgb = (cmap_lut[idx] * hit_any.unsqueeze(-1).float() * 255  # ← * 255
                           ).clamp(0, 255).byte()

            elif mode == "isosurface":
                hit_mask = sampled >= threshold
                hit_idx = hit_mask.long().argmax(dim=0)
                hit_any = hit_mask.any(dim=0)

                hit_grid = torch.stack([
                    sx.gather(0, hit_idx.unsqueeze(0)).squeeze(0),
                    sy.gather(0, hit_idx.unsqueeze(0)).squeeze(0),
                    sz.gather(0, hit_idx.unsqueeze(0)).squeeze(0),
                ], dim=-1).unsqueeze(0).unsqueeze(0)  # (1, 1, out_H, cW, 3)

                grad5 = grad_vol.unsqueeze(0)  # (1, 3, D, H, W)
                norm_xyz = F.grid_sample(
                    grad5, hit_grid,
                    mode="bilinear", padding_mode="zeros", align_corners=True,
                ).squeeze(0).squeeze(1)  # ← squeeze(1) not squeeze(2)
                # (1,3,1,H,W)→(3,1,H,W)→(3,H,W)

                norm_len = norm_xyz.norm(dim=0, keepdim=True).clamp(min=1e-6)
                normal = norm_xyz / norm_len  # (3, out_H, cW)

                light = torch.stack([rx, ry, rz], dim=0)  # (3, out_H, cW)
                diffuse = (normal * light).sum(dim=0).abs()  # (out_H, cW)

                vals = sampled.gather(0, hit_idx.unsqueeze(0)).squeeze(0).clamp(0, 1)
                idx = (vals * (n - 1)).long()

                # cmap_lut already in [0,1] — no /255                ← removed /255
                base_color = cmap_lut[idx]  # (out_H, cW, 3)
                shaded = base_color * (0.3 + 0.7 * diffuse.unsqueeze(-1))
                shaded = shaded * hit_any.unsqueeze(-1).float()
                out_rgb = (shaded * 255).clamp(0, 255).byte()

            elif mode == "depth_color":
                vals, argmax = sampled.max(dim=0)
                depth_norm = argmax.float() / (n_steps - 1)
                vals_g = vals.clamp(0, 1) ** gamma
                depth_idx = (depth_norm * (n - 1)).long()
                # cmap_lut already in [0,1] — no /255               ← removed /255
                out_rgb = (cmap_lut[depth_idx] * vals_g.unsqueeze(-1) * 255
                           ).clamp(0, 255).byte()

            elif mode == "emission_absorption":
                out_rgb = torch.zeros(out_H, cW, 3, device=device)
                T = torch.ones(out_H, cW, device=device)
                step_size = 1.0 / n_steps
                for i in range(n_steps):
                    raw = sampled[i].clamp(0, 1)
                    sigma = raw * opacity_scale
                    alpha = 1.0 - torch.exp(-sigma * step_size)
                    emission = raw ** gamma
                    idx_i = (emission * (n - 1)).long()
                    color = cmap_lut[idx_i]  # ← no /255
                    out_rgb += (T * alpha).unsqueeze(-1) * color
                    T *= torch.exp(-sigma * step_size)
                out_rgb = (out_rgb.clamp(0, 1) * 255).byte()
            elif mode == "alpha_composite":
                out_rgb = torch.zeros(out_H, cW, 3, device=device)
                T = torch.ones(out_H, cW, device=device)
                for i in range(n_steps):
                    alpha = sampled[i].clamp(0, 1) ** gamma
                    color = cmap_lut[(alpha * (n - 1)).long()]
                    w = (T * alpha).unsqueeze(-1)
                    out_rgb += w * color
                    T *= (1.0 - alpha)
                out_rgb = (out_rgb.clamp(0, 1) * 255).byte()  # ← was missing

            else:
                raise ValueError(f"Unknown mode '{mode}'")

                # safety net — should never trigger but prevents UnboundLocalError
            if not isinstance(out_rgb, (torch.Tensor, np.ndarray)):
                raise RuntimeError(f"out_rgb not set for mode '{mode}'")

            out_np[:, col_start:col_end, :] = out_rgb.cpu().numpy()

            del sampled, grid, sx, sy, sz, out_rgb
            torch.cuda.empty_cache()

        return out_np
# def volume_to_equirectangular(
#     volume: torch.Tensor,
#     cmap_lut: torch.Tensor,
#     out_W: int = 2048,
#     n_steps: int = 256,
#     gamma: float = 1.0,
#     col_chunk: int = 256,        # process this many columns at a time — tune down if still OOM
# ) -> np.ndarray:
#     device = volume.device
#     out_H  = out_W // 2
#     n      = cmap_lut.shape[0]
#
#     az_all = torch.linspace(-torch.pi, torch.pi,   out_W, device=device)
#     el_all = torch.linspace( torch.pi/2, -torch.pi/2, out_H, device=device)
#     t      = torch.linspace(0.0, 1.0, n_steps, device=device)  # (n_steps,)
#
#     out_np = np.zeros((out_H, out_W, 3), dtype=np.uint8)
#
#     for col_start in range(0, out_W, col_chunk):
#         col_end = min(col_start + col_chunk, out_W)
#         az = az_all[col_start:col_end]                          # (cW,)
#
#         el_g, az_g = torch.meshgrid(el_all, az, indexing="ij") # (out_H, cW)
#
#         rx =  torch.cos(el_g) * torch.sin(az_g)
#         ry =  torch.sin(el_g)
#         rz =  torch.cos(el_g) * torch.cos(az_g)
#
#         # (n_steps, out_H, cW, 3)
#         sx = rx.unsqueeze(0) * t[:, None, None]
#         sy = ry.unsqueeze(0) * t[:, None, None]
#         sz = rz.unsqueeze(0) * t[:, None, None]
#
#         grid = torch.stack([sx, sy, sz], dim=-1).unsqueeze(0)  # (1, n_steps, out_H, cW, 3)
#
#         vol5    = volume.unsqueeze(0).unsqueeze(0)
#         sampled = F.grid_sample(
#             vol5, grid,
#             mode="bilinear", padding_mode="zeros", align_corners=True,
#         ).squeeze(0).squeeze(0)                                 # (n_steps, out_H, cW)
#
#         # front-to-back composite
#         out_rgb = torch.zeros(out_H, col_end - col_start, 3, device=device)
#         T       = torch.ones (out_H, col_end - col_start,    device=device)
#
#         for i in range(n_steps):
#             alpha    = sampled[i].clamp(0, 1) ** gamma         # (out_H, cW)
#             color    = cmap_lut[(alpha * (n - 1)).long()]       # (out_H, cW, 3)
#             w        = (T * alpha).unsqueeze(-1)
#             out_rgb += w * color
#             T       *= (1.0 - alpha)
#
#         out_np[:, col_start:col_end, :] = (
#             out_rgb.clamp(0, 1).cpu().numpy() * 255
#         ).astype(np.uint8)
#
#         del sampled, grid, sx, sy, sz, out_rgb, T
#         torch.cuda.empty_cache()
#
#     return out_np
# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────
def compile_fractal_step_real(matrix_formula):
    """
    Same as compile_fractal_step but operates on split real/imag tensors.
    torch.compile works fully on real float32 — no fallback warning.
    """
    ops = []
    for i in range(matrix_formula.size[0]):
        row_ops = []
        for j in range(matrix_formula.size[1]):
            item = matrix_formula.matrix[i][j]
            if torch.is_tensor(item):
                row_ops.append(('const',
                                item.real.float().cuda(),
                                item.imag.float().cuda()))
            elif '(' in item:
                minus = -1.0 if '-' in item else 1.0
                fn  = item.split('(')[0].replace('-', '')
                var = item.split('(')[1][:-1]
                row_ops.append(('fn', minus, fn, var))
            else:
                minus = -1.0 if '-' in item else 1.0
                var = item.replace('-', '')
                row_ops.append(('var', minus, var))
        ops.append(row_ops)

    def apply_fn(fn, xr, xi):
        """ Apply fractal function to real/imag pair, return real/imag pair. """
        match fn:
            case 'square':
                return xr*xr - xi*xi, 2*xr*xi
            case 'abs':
                m = torch.sqrt(xr*xr + xi*xi)
                return m, torch.zeros_like(m)
            case 'inverse':
                d = (xr*xr + xi*xi).clamp(min=1e-12)
                return xr/d, -xi/d
            case 'sqrt':
                m = (xr*xr + xi*xi).sqrt().sqrt()
                a = torch.atan2(xi, xr) * 0.5
                return m*torch.cos(a), m*torch.sin(a)
            case 'exp':
                er = torch.exp(xr)
                return er*torch.cos(xi), er*torch.sin(xi)
            case 'log':
                m = (xr*xr + xi*xi).clamp(min=1e-12)
                return 0.5*torch.log(m), torch.atan2(xi, xr)
            case 'sin':
                return torch.sin(xr)*torch.cosh(xi), torch.cos(xr)*torch.sinh(xi)
            case 'cos':
                return torch.cos(xr)*torch.cosh(xi), -torch.sin(xr)*torch.sinh(xi)
            case 'sinh':
                return torch.sinh(xr)*torch.cos(xi), torch.cosh(xr)*torch.sin(xi)
            case 'cosh':
                return torch.cosh(xr)*torch.cos(xi), torch.sinh(xr)*torch.sin(xi)
            case 'tan':
                d = torch.cos(2*xr) + torch.cosh(2*xi)
                return torch.sin(2*xr)/d, torch.sinh(2*xi)/d
            case 'tanh':
                d = torch.cosh(2*xr) + torch.cos(2*xi)
                return torch.sinh(2*xr)/d, torch.sin(2*xi)/d
            case 'angle':
                return torch.atan2(xi, xr), torch.zeros_like(xr)
            case 'sigmoid':
                sr = torch.sigmoid(xr)
                return sr, torch.zeros_like(xi)
            case 'relu':
                return torch.relu(xr), torch.relu(xi)
            case 'rsqrt':
                m = (xr*xr + xi*xi).clamp(min=1e-12)
                return xr / m, -xi / m
            case _:   # sinc, asin, acos, atan, etc — fall back to real part only
                return getattr(torch, fn)(xr), torch.zeros_like(xi)

    def cmul(ar, ai, br, bi):
        return ar*br - ai*bi, ar*bi + ai*br

    def fractal_step(Zr, Zi, Cr, Ci, Dr, Di):
        Rr = torch.zeros_like(Zr)
        Ri = torch.zeros_like(Zi)

        for row_ops in ops:
            def eval_item(op):
                if op[0] == 'const':
                    cr = op[1].expand_as(Zr)
                    ci = op[2].expand_as(Zi)
                    return cr, ci
                elif op[0] == 'fn':
                    _, minus, fn, var = op
                    xr, xi = (Zr,Zi) if var=='Z' else (Cr,Ci) if var=='C' else (Dr,Di)
                    fr, fi = apply_fn(fn, xr, xi)
                    return minus*fr, minus*fi
                else:
                    _, minus, var = op
                    xr, xi = (Zr,Zi) if var=='Z' else (Cr,Ci) if var=='C' else (Dr,Di)
                    return minus*xr, minus*xi

            lr, li = eval_item(row_ops[0])
            for op in row_ops[1:]:
                or2, oi2 = eval_item(op)
                lr, li = cmul(lr, li, or2, oi2)
            Rr = Rr + lr
            Ri = Ri + li

        return Rr, Ri

    return torch.compile(fractal_step)
def voxel_to_vr_video(
        matrix_formula,
        output_path: str = "vr_stereo.mp4",
        colormap: str = "gist_earth",
        eye_disparity: float = 0.03,
        fps: int = 30,
        duration_sec: int = 5,
        equirectangular_mode = {'mode' : 'first_hit', 'chunk' : 256}
):
    """
    Convert a (D, H, W) normalised voxel tensor to a side-by-side stereo MP4.

    Parameters
    ----------
    volume        : torch.Tensor, shape (D, H, W), values in [0, 1]
    output_path   : destination .mp4 file
    colormap      : 'plasma'  or  'viridis'
    eye_disparity : stereo strength in normalised image coords [-1, 1]
                    0.02 = subtle  |  0.04 = comfortable  |  0.08 = strong
    fps           : frames per second
    duration_sec  : length of the video in seconds

    Returns
    -------
    The side-by-side frame as (H, 2*W, 3) uint8 numpy array.
    """
    # if not equirectangular_mode is None:
    #     print("Starting voxel_to_vr_video : Mode =",equirectangular_mode['mode']," Chunks :",equirectangular_mode['chunk'] )
    conf = Configuration()
    out = Path(output_path)
    #
    # volume = torch.flip(volume,(0,2))
    # assert volume.ndim == 3, f"Expected (D, H, W), got {volume.shape}"
    # assert 0.0 <= float(volume.min()) and float(volume.max()) <= 1.0, \
    #     "Values must be in [0, 1]"
    #
    # D, H, W = volume.shape
    # print(f"[voxel_to_vr] volume     : D={D}  H={H}  W={W}")
    # print(f"[voxel_to_vr] colormap   : {colormap}")
    # print(f"[voxel_to_vr] disparity  : ±{eye_disparity}  "
    #       f"({eye_disparity * W:.1f} px at W={W})")

    # cmap_lut = build_cmap_lut(colormap, n=256)
    # ── Step 1 : mean projection  →  (H, W) ──────────────────────────────────
    #proj = mean_projection(volume)

    # ── Step 2 : depth map  →  (H, W) in [0, 1] ──────────────────────────────
    #d_map = depth_map(volume)

    # ── Step 3 : colorise  →  (H, W, 3) uint8 ────────────────────────────────
    # colored = voxel_colored_projection(volume, cmap_lut)
    #colored = depth_colored_projection(volume)
    #colored = apply_colormap(proj, colormap)

    # ── Step 4 : stereo warp ─────────────────────────────────────────────────
    # Left  eye : grid shifted right (+)  → creates outward divergence on near objects
    # Right eye : grid shifted left  (-)
    # left = warp_eye(colored, d_map, +eye_disparity)
    # right = warp_eye(colored, d_map, -eye_disparity)

    # ── Step 5 : side-by-side frame  →  (H, 2W, 3) ───────────────────────────
    # frame = np.concatenate([left, right], axis=1)
    # print(f"[voxel_to_vr] frame size : {frame.shape[1]}x{frame.shape[0]}  (side-by-side)")
    #

    # ── Step 6 : write video (one rotation per frame) ────────────────────────
    n_frames = fps * duration_sec

    #angles = np.linspace(0, 360, n_frames, endpoint=False)  # full spin

    writer = imageio.get_writer(
        str(out),
        fps=fps,
        codec="libx265",  # ← was libx264
        output_params=[
            "-crf", "24",  # H.265 CRF scale is different — 24 ≈ 18 in H.264
            "-preset", "fast",  # slower = smaller file, faster = faster encode
            "-pix_fmt", "yuv420p",
            "-tag:v", "hvc1",  # makes it readable by Apple/Meta players
        ],
    )
    stereo = True
    cmap_lut = build_cmap_lut(colormap)
    out_W = conf.out_W
    col_chunk = 102400

    bounds = find_fractal_bounds(
        matrix_formula,
        currentDi=float(0),
        threshold=4.0,
        max_iter=50,
        search_range=3.0,  # increase to 5.0 or 10.0 if still no hits
        resolution=64,
    )
    if bounds is None:
        return None
    n_steps = 16
    #Di = np.linspace(-bounds['radius'], bounds['radius'], n_frames, endpoint=False)

    time = np.linspace(0, duration_sec, fps*duration_sec, endpoint=False)
    #f0 = .08  # base frequency
    f0 = .05
    Di = np.sin(2 * np.pi * f0 * time)
    Di += 0.5 * np.sin(2 * np.pi * 2 * f0 * time)
    Di += 0.25 * np.sin(2 * np.pi * 3 * f0 * time)
    torch.set_printoptions(profile="full")
    torch.set_printoptions(threshold=100000)  # Set to a number larger than the tensor's total elements
    print(torch.from_numpy(Di))
    #exit(0)
    for i, di in enumerate(Di):

        # Rotate the *view* not the tensor — just rotate the ray directions
        # by building a Y-rotation matrix and applying it to rx,ry,rz
        left = analytical_ray_march_fast(
            compiled_step=matrix_formula,
            currentDi=torch.tensor(di),
            x_difference = eye_disparity,
            out_W=out_W,
            n_steps=n_steps,
            max_iter=conf.max_iter_render,
            threshold=4.0,  # matches your bailouttest
            step_size=bounds['radius'] / n_steps,  # world units per step — tune to your C range
            col_chunk=col_chunk,
            cmap_lut=cmap_lut,
            accumulate=conf.accumulate_rays
        )
        right = analytical_ray_march_fast(
            compiled_step=matrix_formula,
            currentDi=torch.tensor(di),
            x_difference = -eye_disparity,
            out_W=out_W,
            n_steps=n_steps,
            max_iter=conf.max_iter_render,
            threshold=4.0,  # matches your bailouttest
            step_size=bounds['radius'] / n_steps,  # world units per step — tune to your C range
            col_chunk=col_chunk,
            cmap_lut=cmap_lut,
            accumulate=conf.accumulate_rays
        )
        #frame = eq
        # left = warp_eye(eq, ..., +eye_disparity)
        # right = warp_eye(eq, ..., -eye_disparity)
        frame = np.concatenate([left, right], axis=1)
        print("di: ", di)
        writer.append_data(frame)
    # for i, angle in enumerate(angles):
    #     vol_r = rotate_volume_y(volume, angle)
    #     #faces = cubemap_from_center(vol_r, cmap_lut)
    #     d_map = depth_map(volume)
    #
    #
    #     #eq = volume_to_equirectangular(vol_r, cmap_lut, out_W=4096, mode=equirectangular_mode['mode'],col_chunk=equirectangular_mode['chunk'])
    #     eq = volume_to_equirectangular_pointcloud(vol_r, cmap_lut, out_W=4096)
    #     ipd = 2
    #     if stereo:
    #         # Stereo via image-space warp — no black strips possible
    #         left = warp_eye(eq, d_map, +eye_disparity)
    #         right = warp_eye(eq, d_map, -eye_disparity)
    #         # eq_left = volume_to_equirectangular(shift_volume(vol_r, +ipd), cmap_lut, out_W=2048,mode=equirectangular_mode)
    #         # eq_right = volume_to_equirectangular(shift_volume(vol_r, -ipd), cmap_lut, out_W=2048,mode=equirectangular_mode)
    #         frame = np.concatenate([left, right], axis=1)
    #     else:
    #         frame = eq
    #     writer.append_data(frame)

        print(f"[voxel_to_vr] frame {i}/{n_frames}  di={di:.2f}")

    writer.close()
    inject_spatial_metadata(output_path, output_path[0:-4]+"360.mp4", stereo=stereo)
    # ── Step 6 : write video ──────────────────────────────────────────────────
    # n_frames = fps * duration_sec
    # out = Path(output_path)
    # writer = imageio.get_writer(
    #     str(out),
    #     fps=fps,
    #     codec="libx264",
    #     output_params=["-crf", "18", "-pix_fmt", "yuv420p"],
    # )
    # for _ in range(n_frames):
    #     writer.append_data(frame)
    # writer.close()
    #
    # print(f"[voxel_to_vr] saved  →  {out.resolve()}  "
    #       f"({n_frames} frames @ {fps} fps)")
    # return frame
def find_fractal_bounds(
    compiled_step,
    currentDi:   float,
    threshold:   float = 4.0,
    max_iter:    int   = 50,
    search_range: float = 3.0,
    resolution:  int   = 64,     # coarse scan
) -> dict:
    """
    Scans a 3D grid to find where the fractal has surface hits.
    Prints the bounding box so you can calibrate step_size and n_steps.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    coords = torch.linspace(-search_range, search_range, resolution, device=device)
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
    px = gx.reshape(-1)
    py = gy.reshape(-1)
    pz = gz.reshape(-1)
    N  = px.shape[0]

    Zr    = torch.zeros(N, device=device)
    Zi    = torch.zeros(N, device=device)
    Cr    = px.clone()
    Ci    = py.clone()
    Dr    = pz.clone()
    Di    = torch.full((N,), currentDi, device=device)
    alive = torch.ones(N, dtype=torch.bool, device=device)
    bailed = torch.zeros(N, dtype=torch.bool, device=device)

    for i in range(2, max_iter + 1):
        if not alive.any():
            break
        a = alive
        nZr, nZi = compiled_step(Zr[a], Zi[a], Cr[a], Ci[a], Dr[a], Di[a])
        bad = ~torch.isfinite(nZr) | ~torch.isfinite(nZi)
        nZr = torch.where(bad, torch.zeros_like(nZr), nZr)
        nZi = torch.where(bad, torch.zeros_like(nZi), nZi)
        Zr[a], Zi[a] = nZr, nZi

        bad_full = torch.zeros_like(alive)
        bad_full[a] = bad
        hit = ((Zr + Zi > threshold) | bad_full) & alive
        bailed[hit] = True
        Zr[hit]     = 0.0
        Zi[hit]     = 0.0
        alive[hit]  = False

    if not bailed.any():
        print("[find_fractal_bounds] NO HITS FOUND in range ±{search_range}")
        print("  → Try increasing search_range or lowering threshold")
        return None

    hit_px = px[bailed]
    hit_py = py[bailed]
    hit_pz = pz[bailed]

    bounds = {
        "x": (float(hit_px.min()), float(hit_px.max())),
        "y": (float(hit_py.min()), float(hit_py.max())),
        "z": (float(hit_pz.min()), float(hit_pz.max())),
        "center_x": float((hit_px.min() + hit_px.max()) / 2),
        "center_y": float((hit_py.min() + hit_py.max()) / 2),
        "center_z": float((hit_pz.min() + hit_pz.max()) / 2),
        "radius":   float(torch.stack([hit_px, hit_py, hit_pz]).norm(dim=0).max()),
        "n_hits":   int(bailed.sum()),
        "hit_rate": float(bailed.sum()) / N,
    }

    print(f"[find_fractal_bounds] Found {bounds['n_hits']} hits "
          f"({bounds['hit_rate']*100:.1f}% of grid)")
    print(f"  x range : {bounds['x'][0]:.3f} → {bounds['x'][1]:.3f}")
    print(f"  y range : {bounds['y'][0]:.3f} → {bounds['y'][1]:.3f}")
    print(f"  z range : {bounds['z'][0]:.3f} → {bounds['z'][1]:.3f}")
    print(f"  radius  : {bounds['radius']:.3f}")
    print(f"\n  Suggested ray march parameters:")
    r = bounds['radius']
    print(f"    step_size = {r / 64:.4f}   # radius / n_steps")
    print(f"    n_steps   = 64")
    print(f"    → max_t   = {r / 64 * 64:.3f}  (covers full diameter)")

    return bounds
def analytical_ray_march_fast(
    compiled_step,
    currentDi:    torch.Tensor,
    out_W:        int   = 2048,
    x_difference: float = 0,
    n_steps:      int   = 64,
    max_iter:     int   = 50,
    threshold:    float = 4.0,
    step_size:    float = 0.03,
    col_chunk:    int   = 128,
    step_chunk:   int   = 8,
    cmap_lut:     torch.Tensor = None,
    accumulate:   bool  = True,
    gamma:        float = 0.5,
) -> np.ndarray:
    """
    No alive mask — all points iterated unconditionally every step.
    Reduces Python iterations from max_iter×n_kernel_launches to max_iter total.
    """
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_H   = out_W // 2
    n       = cmap_lut.shape[0]
    Di_val  = float(currentDi)

    az_all = torch.linspace(-torch.pi, torch.pi,     out_W, device=device)
    el_all = torch.linspace( torch.pi/2, -torch.pi/2, out_H, device=device)
    t_vals = torch.arange(1, n_steps + 1, device=device, dtype=torch.float32) * step_size

    out_np = np.zeros((out_H, out_W, 3), dtype=np.uint8)

    def stretch(t, mask):
        if not mask.any():
            return t
        lo = t[mask].min()
        hi = t[mask].max().clamp(min=lo + 1e-8)
        return ((t - lo) / (hi - lo)).clamp(0, 1)

    for col_start in range(0, out_W, col_chunk):
        col_end = min(col_start + col_chunk, out_W)
        cW      = col_end - col_start
        az      = az_all[col_start:col_end]

        el_g, az_g = torch.meshgrid(el_all, az, indexing="ij")

        rx_base = (torch.cos(el_g) * torch.sin(az_g)) + x_difference
        ry_base =  torch.sin(el_g)
        rz_base =  torch.cos(el_g) * torch.cos(az_g)

        total_bail  = torch.zeros(out_H, cW, device=device)
        total_mag   = torch.zeros(out_H, cW, device=device)
        total_count = torch.zeros(out_H, cW, device=device)

        for s_start in range(0, n_steps, step_chunk):
            s_end = min(s_start + step_chunk, n_steps)
            sC    = s_end - s_start
            t_sl  = t_vals[s_start:s_end]

            px = (rx_base.unsqueeze(0) * t_sl[:, None, None]).reshape(-1)
            py = (ry_base.unsqueeze(0) * t_sl[:, None, None]).reshape(-1)
            pz = (rz_base.unsqueeze(0) * t_sl[:, None, None]).reshape(-1)
            N  = px.shape[0]

            Zr = torch.zeros(N, device=device)
            Zi = torch.zeros(N, device=device)
            Cr = px
            Ci = py
            Dr = pz
            Di = torch.full((N,), Di_val, device=device)

            # No alive mask — these accumulate hits across all iterations
            bail_acc  = torch.zeros(N, device=device)
            mag_acc   = torch.zeros(N, device=device)
            hit_count = torch.zeros(N, device=device)
            # frozen: points that bailed get Z zeroed and stay zeroed
            frozen    = torch.zeros(N, dtype=torch.bool, device=device)

            for i in range(2, max_iter + 1):
                # ── No masking — full tensor op every iteration ───────────
                nZr, nZi = compiled_step(Zr, Zi, Cr, Ci, Dr, Di)

                # Sanitize in-place — no new tensor allocation
                bad = ~torch.isfinite(nZr) | ~torch.isfinite(nZi)
                nZr.nan_to_num_(0.0)
                nZi.nan_to_num_(0.0)

                # Zero out frozen points so they don't affect bailout test
                nZr[frozen] = 0.0
                nZi[frozen] = 0.0
                Zr = nZr
                Zi = nZi

                # Bailout test — pure tensor, no Python branching
                bailed = (Zr + Zi > threshold) | (bad & ~frozen)

                # Accumulate unconditionally with .where — no indexing
                new_mag = (Zr.pow(2) + Zi.pow(2)).sqrt()
                bail_acc  = torch.where(bailed, bail_acc  + i,        bail_acc)
                mag_acc   = torch.where(bailed, mag_acc   + new_mag,  mag_acc)
                hit_count = torch.where(bailed, hit_count + 1,        hit_count)

                # Reset bailed points
                Zr = torch.where(bailed, torch.zeros_like(Zr), Zr)
                Zi = torch.where(bailed, torch.zeros_like(Zi), Zi)

                if not accumulate:
                    frozen = frozen | bailed   # stop accumulating after first hit

            bail_3d  = bail_acc .reshape(sC, out_H, cW)
            mag_3d   = mag_acc  .reshape(sC, out_H, cW)
            count_3d = hit_count.reshape(sC, out_H, cW)

            total_bail  += bail_3d .max(dim=0).values
            total_mag   += mag_3d  .max(dim=0).values
            total_count += count_3d.sum(dim=0)

            del px, py, pz, Zr, Zi, Cr, Ci, Dr, Di
            del bail_acc, mag_acc, hit_count, frozen
            del bail_3d, mag_3d, count_3d
            torch.cuda.empty_cache()

        # ── Colour ────────────────────────────────────────────────────────
        hit_any  = total_count > 0
        density  = (total_count / (n_steps * max_iter)).clamp(0, 1)

        mag_norm  = stretch(total_mag,  hit_any)
        bail_norm = stretch(total_bail, hit_any)

        if accumulate:
            value = (mag_norm * 0.4 + bail_norm * 0.4 + density * 0.2) * hit_any.float()
        else:
            value = ((mag_norm + bail_norm) / 2) * hit_any.float()

        value   = torch.nan_to_num(value, nan=0.0, posinf=1.0, neginf=0.0).pow(gamma)
        lut_idx = (value * (n - 1)).long().clamp(0, n - 1)
        out_rgb = (
            cmap_lut[lut_idx] * hit_any.float().unsqueeze(-1) * 255
        ).clamp(0, 255).byte()

        out_np[:, col_start:col_end, :] = out_rgb.reshape(out_H, cW, 3).cpu().numpy()

        del total_bail, total_mag, total_count, rx_base, ry_base, rz_base
        torch.cuda.empty_cache()

    return out_np

def analytical_ray_march_equirectangular(
    matrix_formula,          # your MatrixFormula instance
    out_W:      int   = 2048,
    n_steps:    int   = 64,   # steps per ray — much less needed than grid sampling
    max_iter:   int   = 50,   # fractal iterations per sample point
    threshold:  float = 4.0,  # bailout condition  (real+imag > 4 in your code)
    step_size:  float = 0.03, # world-space step along each ray
    col_chunk:  int   = 128,  # columns per batch — tune for VRAM
    cmap_lut:   torch.Tensor  = None,
) -> np.ndarray:
    """
    Fully analytical ray marching — no precomputed tensor.
    For each ray, evaluates the fractal formula directly at 3D sample points.

    C = px + i*py  (the complex coordinate plane)
    D = pz + 0j    (depth maps to the time/D axis)
    Z starts at 0  (reset at each sample point — surface test)

    Returns (out_H, out_W, 3) uint8 equirectangular image.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_H  = out_W // 2
    n      = cmap_lut.shape[0]

    az_all = torch.linspace(-torch.pi, torch.pi,    out_W, device=device)
    el_all = torch.linspace( torch.pi/2, -torch.pi/2, out_H, device=device)

    out_np = np.zeros((out_H, out_W, 3), dtype=np.uint8)

    for col_start in range(0, out_W, col_chunk):
        col_end = min(col_start + col_chunk, out_W)
        cW      = col_end - col_start
        az      = az_all[col_start:col_end]

        el_g, az_g = torch.meshgrid(el_all, az, indexing="ij")  # (out_H, cW)

        # Ray directions — unit vectors on the sphere
        rx = (torch.cos(el_g) * torch.sin(az_g)).flatten()   # (out_H*cW,)
        ry = (torch.sin(el_g)).flatten()
        rz = (torch.cos(el_g) * torch.cos(az_g)).flatten()
        n_rays = rx.shape[0]

        # Accumulators
        hit_color    = torch.zeros(n_rays, 3, device=device)
        hit_mask     = torch.zeros(n_rays, dtype=torch.bool, device=device)

        # March all rays simultaneously
        for step in range(n_steps):
            # Skip rays that already hit something
            active = ~hit_mask                                   # (n_rays,)
            if not active.any():
                break

            t = step * step_size

            # 3D sample point for all active rays
            px = rx[active] * t    # (n_active,)
            py = ry[active] * t
            pz = rz[active] * t

            # ── Map to fractal variables ───────────────────────────────────
            # This is the key: we inject 3D coordinates into C and D
            n_active = px.shape[0]

            # Create a lightweight proxy object so process_matrix can call
            # get_C_var(), get_D_var(), get_Z_var()
            class RayProxy:
                def __init__(self):
                    self.window_C = torch.complex(px, py).to(device)
                    self.window_D = torch.complex(pz, torch.zeros_like(pz)).to(device)
                    self.window_Z = torch.zeros(n_active, dtype=torch.complex64, device=device)
                def get_C_var(self): return self.window_C
                def get_D_var(self): return self.window_D
                def get_Z_var(self): return self.window_Z
                def get_empty_window(self, dtype=torch.complex64):
                    return torch.zeros(n_active, dtype=dtype, device=device)

            proxy = RayProxy()

            # ── Iterate the fractal formula at these points ────────────────
            bailout_iter = torch.zeros(n_active, device=device)
            exit_Z       = torch.zeros(n_active, dtype=torch.complex64, device=device)
            still_alive  = torch.ones(n_active, dtype=torch.bool, device=device)

            for i in range(2, max_iter + 1):
                proxy.window_Z = matrix_formula.process_matrix(proxy)

                # Bailout test  (same as your bailouttest)
                bailed = (proxy.window_Z.real + proxy.window_Z.imag > threshold) & still_alive
                bailout_iter[bailed] = i
                exit_Z[bailed]       = proxy.window_Z[bailed]
                proxy.window_Z[bailed] = float('nan') + float('nan') * 1j
                still_alive[bailed]  = False

                if not still_alive.any():
                    break

            # ── Points that bailed out = on or near the surface ───────────
            hit_here = bailout_iter > 0                          # (n_active,)

            if hit_here.any():
                # Colour using normalised bailout + exit magnitude
                # (mirrors your prepare_tensor_for_video logic)
                magnitude    = exit_Z[hit_here].abs()
                mag_norm     = (magnitude / (magnitude.max().clamp(min=1e-8)))
                bail_norm    = bailout_iter[hit_here] / max_iter
                value        = ((mag_norm + bail_norm) / 2).clamp(0, 1)
                idx          = (value * (n - 1)).long()
                colors       = cmap_lut[idx]                     # (n_hit, 3)

                # Write into the full ray buffer
                active_indices        = active.nonzero(as_tuple=True)[0]
                hit_indices           = active_indices[hit_here]
                hit_color[hit_indices] = colors
                hit_mask[hit_indices]  = True

        # Reshape back to (out_H, cW, 3)
        out_rgb = (hit_color.reshape(out_H, cW, 3) * 255).clamp(0, 255).byte()
        out_np[:, col_start:col_end, :] = out_rgb.cpu().numpy()

        del rx, ry, rz, hit_color, hit_mask
        torch.cuda.empty_cache()

    return out_np

def voxel_to_vr_video_hires(
    volume,
    output_path   = "vr_hires.mp4",
    colormap      = "gist_earth",
    eye_disparity = 0.06,
    fps           = 30,
    duration_sec  = 10,
    out_W         = 4096,
    threshold     = 0.15,
    base_radius   = 2.0,
    max_radius    = 5,
    max_points    = 500000,
    gamma         = 1.0,
):
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    volume   = volume.to(device)
    cmap_lut = build_cmap_lut(colormap)
    out_H    = out_W // 2
    n        = cmap_lut.shape[0]
    D, H, W  = volume.shape

    # ── Extract points once ───────────────────────────────────────────────────
    vol_cpu = volume.cpu()
    d_idx, h_idx, w_idx = (vol_cpu >= threshold).nonzero(as_tuple=True)
    vals = vol_cpu[d_idx, h_idx, w_idx]
    print(f"[hires] active voxels: {len(vals):,}")

    if len(vals) > max_points:
        step  = max(1, len(vals) // max_points)
        d_idx, h_idx, w_idx = d_idx[::step], h_idx[::step], w_idx[::step]
        vals  = vals[::step]
    print(f"[hires] rendering points: {len(vals):,}")

    x0 =  (w_idx.float() / (W-1)) * 2 - 1
    y0 = -((h_idx.float() / (H-1)) * 2 - 1)
    z0 =  (d_idx.float() / (D-1)) * 2 - 1
    xyz0   = torch.stack([x0, y0, z0], dim=-1).to(device)
    cidx   = (vals.clamp(0,1) ** gamma * (n-1)).long().to(device)
    colors = cmap_lut[cidx]

    # Disk offsets precomputed once
    r_int   = int(max_radius)
    offsets = [(dy, dx)
               for dy in range(-r_int, r_int+1)
               for dx in range(-r_int, r_int+1)
               if dx*dx + dy*dy <= r_int*r_int]

    angles  = np.linspace(0, 360, fps * duration_sec, endpoint=False)
    writer = imageio.get_writer(
        str(output_path),
        fps=fps,
        codec="libx265",  # ← was libx264
        output_params=[
            "-crf", "24",  # H.265 CRF scale is different — 24 ≈ 18 in H.264
            "-preset", "fast",  # slower = smaller file, faster = faster encode
            "-pix_fmt", "yuv420p",
            "-tag:v", "hvc1",  # makes it readable by Apple/Meta players
        ],
    )

    for i, angle in enumerate(angles):
        # Rotate coords — no volume resample
        R   = make_rotation_matrix_y(angle, device)
        xyz = (R @ xyz0.T).T
        x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
        dist = (x**2 + y**2 + z**2).sqrt().clamp(min=1e-6)

        # Far → near sort
        order          = dist.argsort(descending=True)
        x, y, z        = x[order], y[order], z[order]
        dist, c_sorted = dist[order], colors[order]

        # Project
        az = torch.atan2(x, z)
        el = torch.asin((y / dist).clamp(-1, 1))
        px = ((az / (2*torch.pi) + 0.5) * out_W).long().clamp(0, out_W-1)
        py = ((0.5 - el / torch.pi) * out_H).long().clamp(0, out_H-1)

        radius = (base_radius / dist).clamp(1, max_radius)

        # Splat both eyes simultaneously on one wide canvas (out_H × 2*out_W)
        # Left eye: normal projection
        # Right eye: shift az by +eye_disparity before projecting
        def splat(px_, py_, cols):
            canvas = torch.zeros(out_H * out_W, 3, device=device)
            for dy, dx in offsets:
                d2       = dx*dx + dy*dy
                in_splat = (radius*radius) >= d2
                if not in_splat.any(): continue
                ty   = (py_[in_splat] + dy).clamp(0, out_H-1)
                tx   = (px_[in_splat] + dx).clamp(0, out_W-1)
                canvas.index_put_((ty*out_W+tx,), cols[in_splat], accumulate=False)
            return canvas.view(out_H, out_W, 3)

        left_canvas  = splat(px, py, c_sorted)

        # Right eye: tiny az shift for stereo
        az_r  = az + eye_disparity
        px_r  = ((az_r / (2*torch.pi) + 0.5) * out_W).long().clamp(0, out_W-1)
        right_canvas = splat(px_r, py, c_sorted)

        frame = torch.cat([left_canvas, right_canvas], dim=1)
        writer.append_data(
            (frame.clamp(0,1).cpu().numpy() * 255).astype(np.uint8)
        )

        #if i % fps == 0:
        print(f"[hires] frame {i}/{len(angles)}  angle={angle:.1f}°")

    writer.close()
    inject_spatial_metadata(output_path,
                            output_path.replace(".mp4", "_360.mp4"),
                            stereo=True)
# ──────────────────────────────────────────────────────────────────────────────
# Demo  —  replace volume with your own tensor
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Swap this block for your real (D, H, W) tensor ───────────────────────
    D, H, W = 128, 512, 512
    zz, yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, D),
        torch.linspace(-1, 1, H),
        torch.linspace(-1, 1, W),
        indexing="ij",
    )
    volume = torch.exp(-(zz ** 2 + yy ** 2 + xx ** 2) / 0.3)
    volume = (volume - volume.min()) / (volume.max() - volume.min())
    # ─────────────────────────────────────────────────────────────────────────

    voxel_to_vr_video(
        volume,
        output_path="vr_stereo.mp4",
        colormap="plasma",  # or "viridis"
        eye_disparity=0.2,  # tune for your VR headset  (0.02 – 0.08)
        fps=30,
        duration_sec=5,
    )