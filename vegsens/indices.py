import numpy as np
EPS = 1e-6
DEFAULT_W = (0.40, 0.25, 0.20, 0.15)

def indices(img):
    R, G, B, N = (img[..., i] for i in range(4))
    ndvi = (N - R) / (N + R + EPS)
    den = G + R - B
    vari = np.where(np.abs(den) < 1e-3, 0.0, (G - R) / np.where(np.abs(den) < 1e-3, 1, den))
    exg = 2 * G - R - B
    gli = (2 * G - R - B) / (2 * G + R + B + EPS)
    return dict(NDVI=ndvi, VARI=vari, ExG=exg, GLI=gli)

def minmax(u, robust=True):
    lo, hi = (np.percentile(u, 1), np.percentile(u, 99)) if robust else (u.min(), u.max())
    return np.clip((u - lo) / (hi - lo + EPS), 0, 1)

def ess(idx, w=DEFAULT_W):
    w = np.asarray(w, float); w = w / w.sum()
    return sum(wi * minmax(idx[k]) for wi, k in zip(w, ["NDVI", "VARI", "ExG", "GLI"]))

def slope(dem):
    gy, gx = np.gradient(dem)
    return np.arctan(np.hypot(gx, gy))

def cost_surface(E, c0=0.1, alpha=1.2, beta=0.0, gamma=0.0, delta=0.0, slope_n=None, prot=None, flood=None):
    C = c0 + alpha * E
    if beta: C = C + beta * slope_n
    if gamma: C = C + gamma * prot
    if delta: C = C + delta * flood
    return C
