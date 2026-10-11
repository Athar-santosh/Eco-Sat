"""Procedural generator for synthetic four-band (R, G, B, NIR) test scenes with
known-by-construction five-class labels (0 water, 1 urban, 2 barren, 3 sparse, 4 dense)."""
import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt

# class-mean surface reflectance (R, G, B, NIR)
REFLECTANCE = np.array([
    [0.05, 0.07, 0.12, 0.03],   # water
    [0.25, 0.25, 0.27, 0.30],   # urban
    [0.35, 0.30, 0.22, 0.38],   # barren
    [0.12, 0.15, 0.08, 0.35],   # sparse vegetation
    [0.04, 0.09, 0.04, 0.50],   # dense vegetation
])
NOISE_CORR_AMP, NOISE_CORR_SIGMA, NOISE_WHITE = 0.02, 3.0, 0.03

# scene-specific parameters: number of sparse-vegetation clearings in the forest
# (fewer clearings -> denser vegetation) and generator seed
SCENES = {"S1": dict(n_clear=26, seed=101), "S2": dict(n_clear=18, seed=202),
          "S3": dict(n_clear=11, seed=303), "S4": dict(n_clear=6, seed=404),
          "S5": dict(n_clear=2, seed=505)}


def make_scene(name="S3", size=220, n_clear=None, seed=None):
    p = SCENES.get(name, {})
    n_clear = p.get("n_clear", 11) if n_clear is None else n_clear
    seed = p.get("seed", 0) if seed is None else seed
    rng = np.random.default_rng(seed)
    H = W = size
    s = size / 220.0
    yy, xx = np.mgrid[0:H, 0:W]
    lab = np.full((H, W), 4, np.int8)                       # dense background
    # sparse-vegetation clearings (smooth blobs)
    for _ in range(n_clear):
        cy, cx = rng.uniform(0, H), rng.uniform(0, W)
        ry, rx = rng.uniform(8, 22) * s, rng.uniform(8, 22) * s
        lab[((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1] = 3
    # central low-vegetation corridor (gently undulating band)
    yc = 0.5 * H + 6 * s * np.sin(2 * np.pi * xx / (W / 1.5) + rng.uniform(0, 6))
    lab[np.abs(yy - yc) <= 10 * s] = 3
    # barren and urban blocks
    by, bx = int(0.70 * H), int(0.12 * W)
    lab[by:by + int(0.18 * H), bx:bx + int(0.22 * W)] = 2
    uy, ux = int(0.10 * H), int(0.68 * W)
    lab[uy:uy + int(0.16 * H), ux:ux + int(0.18 * W)] = 1
    # sinuous river
    xr = 0.5 * W + 0.33 * W * np.sin(2 * np.pi * yy / (H * 0.9) + rng.uniform(0, 6)) \
        + 0.08 * W * np.sin(2 * np.pi * yy / (H * 0.3))
    lab[np.abs(xx - xr) <= 3.5 * s] = 0
    # reflectance with spatially correlated + white noise
    img = REFLECTANCE[lab].astype(np.float64)
    for b in range(4):
        img[..., b] += NOISE_CORR_AMP * gaussian_filter(rng.standard_normal((H, W)), NOISE_CORR_SIGMA) * NOISE_CORR_SIGMA
        img[..., b] += NOISE_WHITE * rng.standard_normal((H, W))
    img = np.clip(img, 0.001, 1.0)
    # auxiliary layers: DEM with a ridge, protected polygon, flood buffer around water
    dem = 80 * np.exp(-((xx - 0.30 * W) ** 2) / (2 * (0.07 * W) ** 2)) * (0.6 + 0.4 * np.sin(yy / H * np.pi))
    dem += 5 * gaussian_filter(rng.standard_normal((H, W)), 8) * 8
    prot = np.zeros((H, W)); prot[int(0.30 * H):int(0.46 * H), int(0.42 * W):int(0.60 * W)] = 1
    flood = (distance_transform_edt(lab != 0) <= 8 * s).astype(float)
    start, goal = (int(0.18 * H), 0), (int(0.82 * H), W - 1)
    return dict(name=name, img=img, label=lab, dem=dem, prot=prot, flood=flood,
                start=start, goal=goal, seed=seed, n_clear=n_clear)
