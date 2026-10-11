"""Exact least-cost search on an 8-connected grid (A*, Dijkstra) with numba.
Edge weight w(x,x') = 0.5 (C(x)+C(x')) d(x,x'), d in {1, sqrt 2} (Eq. 16)."""
import numpy as np
from numba import njit

DR = np.array([-1, -1, -1, 0, 0, 1, 1, 1]); DC = np.array([-1, 0, 1, -1, 1, -1, 0, 1])
DD = np.array([np.sqrt(2), 1, np.sqrt(2), 1, 1, np.sqrt(2), 1, np.sqrt(2)])

@njit(cache=True)
def _push(hk, hv, n, key, val):
    i = n; hk[i] = key; hv[i] = val
    while i > 0:
        p = (i - 1) >> 1
        if hk[p] <= hk[i]: break
        hk[p], hk[i] = hk[i], hk[p]; hv[p], hv[i] = hv[i], hv[p]; i = p
    return n + 1

@njit(cache=True)
def _pop(hk, hv, n):
    k0, v0 = hk[0], hv[0]; n -= 1
    hk[0] = hk[n]; hv[0] = hv[n]; i = 0
    while True:
        l = 2 * i + 1; r = l + 1; m = i
        if l < n and hk[l] < hk[m]: m = l
        if r < n and hk[r] < hk[m]: m = r
        if m == i: break
        hk[m], hk[i] = hk[i], hk[m]; hv[m], hv[i] = hv[i], hv[m]; i = m
    return k0, v0, n

@njit(cache=True)
def _search(C, sr, sc, gr, gc, hscale, r0, r1, c0, c1, DR, DC, DD):
    H, W = C.shape
    g = np.full(H * W, np.inf); par = np.full(H * W, -1, np.int64); closed = np.zeros(H * W, np.bool_)
    cap = 8 * (r1 - r0 + 1) * (c1 - c0 + 1) + 16
    hk = np.empty(cap); hv = np.empty(cap, np.int64); n = 0
    s = sr * W + sc; t = gr * W + gc; g[s] = 0.0
    n = _push(hk, hv, n, hscale * np.hypot(sr - gr, sc - gc), s)
    expanded = 0
    while n > 0:
        f, u, n = _pop(hk, hv, n)
        if closed[u]: continue
        closed[u] = True; expanded += 1
        if u == t: break
        ur, uc = u // W, u % W
        for k in range(8):
            vr, vc = ur + DR[k], uc + DC[k]
            if vr < r0 or vr > r1 or vc < c0 or vc > c1: continue
            v = vr * W + vc
            if closed[v]: continue
            nd = g[u] + 0.5 * (C[ur, uc] + C[vr, vc]) * DD[k]
            if nd < g[v]:
                g[v] = nd; par[v] = u
                if n >= cap: return np.empty(0, np.int64), np.inf, expanded
                n = _push(hk, hv, n, nd + hscale * np.hypot(vr - gr, vc - gc), v)
    if par[t] == -1 and s != t: return np.empty(0, np.int64), np.inf, expanded
    path = [t]
    while path[-1] != s: path.append(par[path[-1]])
    out = np.empty(len(path), np.int64)
    for i in range(len(path)): out[i] = path[len(path) - 1 - i]
    return out, g[t], expanded

def shortest_path(C, s, t, heuristic=True, box=None, margin=None):
    """A* (heuristic=True, admissible h = min(C)*Euclidean) or Dijkstra (h = 0). Returns list of (r,c), cost, expanded."""
    H, W = C.shape
    if box is None:
        if margin is None: r0, r1, c0, c1 = 0, H - 1, 0, W - 1
        else:
            r0 = max(0, min(s[0], t[0]) - margin); r1 = min(H - 1, max(s[0], t[0]) + margin)
            c0 = max(0, min(s[1], t[1]) - margin); c1 = min(W - 1, max(s[1], t[1]) + margin)
    else: r0, r1, c0, c1 = box
    hs = float(C.min()) if heuristic else 0.0
    p, cost, ex = _search(np.ascontiguousarray(C, np.float64), s[0], s[1], t[0], t[1], hs, r0, r1, c0, c1, DR, DC, DD)
    return [(int(v // W), int(v % W)) for v in p], float(cost), int(ex)
