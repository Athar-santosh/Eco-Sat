import numpy as np
from . import DENSE

def path_arrays(P):
    a = np.asarray(P); return a[:, 0], a[:, 1]

def f_len(P):
    a = np.asarray(P, float); d = np.abs(np.diff(a, axis=0)).sum(1)
    return float(np.where(d == 2, np.sqrt(2), d).sum())

def f_eco(P, E):
    r, c = path_arrays(P); return float(E[r[1:], c[1:]].mean())

def a_eco(P, E):
    a = np.asarray(P, float); d = np.abs(np.diff(a, axis=0)).sum(1); d = np.where(d == 2, np.sqrt(2), d)
    r, c = path_arrays(P); return float((d * E[r[1:], c[1:]]).sum())

def route_metrics(P, E, ndvi, lab, J=None):
    r, c = path_arrays(P)
    return dict(length=f_len(P), f_eco=f_eco(P, E), a_eco=a_eco(P, E), ndvi=float(ndvi[r[1:], c[1:]].mean()),
                dense=int((lab[r, c] == DENSE).sum()), m=len(P) - 1)

def hypervolume_2d(F, ref=(1.1, 1.1)):
    """F: (n,2) minimisation points (already normalised)."""
    F = np.asarray(F, float); F = F[(F[:, 0] < ref[0]) & (F[:, 1] < ref[1])]
    if len(F) == 0: return 0.0
    F = F[np.argsort(F[:, 0])]; hv, best = 0.0, ref[1]
    for x, y in F:
        if y < best: hv += (ref[0] - x) * (best - y); best = y
    return float(hv)

def nondominated(F):
    F = np.asarray(F); keep = np.ones(len(F), bool)
    for i in range(len(F)):
        if keep[i]:
            dom = np.all(F <= F[i], 1) & np.any(F < F[i], 1)
            if dom.any(): keep[i] = False
    return keep
