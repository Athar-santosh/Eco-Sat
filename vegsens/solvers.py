"""Standard solvers applied without problem-specific modification:
A*, Dijkstra (exact, scalarised cost J), eco-aware ACO, NSGA-II, and the A*-seeded memetic NSGA-II benchmark."""
import numpy as np
from numba import njit
from .graphsearch import shortest_path, DR, DC, DD
from .metrics import f_len, f_eco, a_eco

# ---------------- exact ----------------
def astar(C, s, g):
    P, J, ex = shortest_path(C, s, g, heuristic=True); return P, dict(J=J, expanded=ex)

def dijkstra(C, s, g):
    P, J, ex = shortest_path(C, s, g, heuristic=False); return P, dict(J=J, expanded=ex)

# ---------------- ACO ----------------
@njit(cache=True)
def _aco(C, dense, sr, sc, gr, gc, n_ants, n_iter, a, b, rho, Q, pen, seed, DR, DC, DD, maxf):
    np.random.seed(seed)
    H, W = C.shape; tau = np.ones((H, W, 8)); c0 = C.min()
    best_J = np.inf; best = np.empty(0, np.int64)
    maxs = maxf * (H + W)
    visited = np.zeros(H * W, np.int32); stamp = 0
    for it in range(n_iter):
        paths = []; Js = []
        for ant in range(n_ants):
            stamp += 1
            r, c = sr, sc; path = [r * W + c]; visited[r * W + c] = stamp; J = 0.0; ok = False
            for step in range(maxs):
                if r == gr and c == gc: ok = True; break
                probs = np.zeros(8); hu = np.hypot(r - gr, c - gc)
                for k in range(8):
                    vr, vc = r + DR[k], c + DC[k]
                    if vr < 0 or vr >= H or vc < 0 or vc >= W: continue
                    if visited[vr * W + vc] == stamp: continue
                    w = 0.5 * (C[r, c] + C[vr, vc]) * DD[k]
                    red = w + c0 * (np.hypot(vr - gr, vc - gc) - hu)   # goal-directed reduced cost >= 0
                    eta = 1.0 / (red + 1e-3)
                    if dense[vr, vc]: eta /= pen
                    probs[k] = tau[r, c, k] ** a * eta ** b
                tot = probs.sum()
                if tot <= 0:                      # dead end: backtrack one step (cell stays tabu)
                    if len(path) <= 1: break
                    path.pop(); last = path[-1]; r, c = last // W, last % W
                    continue
                u = np.random.random() * tot; acc = 0.0; kk = 7
                for k in range(8):
                    acc += probs[k]
                    if acc >= u and probs[k] > 0: kk = k; break
                vr, vc = r + DR[kk], c + DC[kk]
                r, c = vr, vc; path.append(r * W + c); visited[r * W + c] = stamp
            if ok:
                arr = np.array(path); J = 0.0
                for q in range(len(arr) - 1):
                    a0, b0 = arr[q], arr[q + 1]; d = 1.4142135623730951 if (a0 // W != b0 // W and a0 % W != b0 % W) else 1.0
                    J += 0.5 * (C[a0 // W, a0 % W] + C[b0 // W, b0 % W]) * d
                paths.append(arr); Js.append(J)
                if J < best_J: best_J = J; best = arr
        tau *= (1 - rho)
        for i in range(len(paths)):
            p = paths[i]
            for j in range(len(p) - 1):
                u0, v0 = p[j], p[j + 1]; dr = v0 // W - u0 // W; dc = v0 % W - u0 % W
                for k in range(8):
                    if DR[k] == dr and DC[k] == dc: tau[u0 // W, u0 % W, k] += Q / Js[i]
        for j in range(len(best) - 1):   # elitist reinforcement of the global best
            u0, v0 = best[j], best[j + 1]; dr = v0 // W - u0 // W; dc = v0 % W - u0 % W
            for k in range(8):
                if DR[k] == dr and DC[k] == dc: tau[u0 // W, u0 % W, k] += Q / best_J
    return best, best_J

def aco(C, s, g, dense_mask, n_ants=28, n_iter=40, a=1.0, b=2.0, rho=0.1, Q=1.0, pen=5.0, seed=0, max_steps_factor=4):
    p, J = _aco(np.ascontiguousarray(C), dense_mask.astype(np.bool_), s[0], s[1], g[0], g[1],
                n_ants, n_iter, a, b, rho, Q, pen, seed, DR, DC, DD, max_steps_factor)
    W = C.shape[1]
    return [(int(v // W), int(v % W)) for v in p], dict(J=float(J))

# ---------------- NSGA-II / memetic ----------------
def remove_loops(P):
    out, pos = [], {}
    for v in P:
        if v in pos:
            k = pos[v]
            for u in out[k + 1:]: pos.pop(u, None)
            out = out[:k + 1]
        else:
            pos[v] = len(out); out.append(v)
    return out

def crossover(p1, p2, rng):
    common = set(p1[1:-1]) & set(p2[1:-1])
    if not common: return list(p1), list(p2)
    v = list(common)[rng.integers(len(common))]
    i, j = p1.index(v), p2.index(v)
    return remove_loops(p1[:i] + p2[j:]), remove_loops(p2[:j] + p1[i:])

def mutate(P, C, rng, lmax=30, delta=8):
    if len(P) < 6: return P
    H, W = C.shape
    i = int(rng.integers(0, len(P) - 3)); j = min(len(P) - 1, i + int(rng.integers(4, lmax + 1)))
    k = (i + j) // 2
    dy, dx = P[j][0] - P[i][0], P[j][1] - P[i][1]; nrm = np.hypot(dy, dx) or 1.0
    off = rng.integers(-delta, delta + 1)
    wy = int(np.clip(round(P[k][0] - dx / nrm * off), 0, H - 1)); wx = int(np.clip(round(P[k][1] + dy / nrm * off), 0, W - 1))
    a, _, _ = shortest_path(C, P[i], (wy, wx), margin=10)
    b, _, _ = shortest_path(C, (wy, wx), P[j], margin=10)
    if not a or not b: return P          # repair failed -> keep parent
    return remove_loops(P[:i] + a + b[1:] + P[j + 1:])

def local_search(P, E, Ebar, lam, Lref, rng, L=12):
    if len(P) < L + 2: return P
    i = int(rng.integers(0, len(P) - L)); j = i + L
    Cl = lam * E + (1 - lam) * Ebar
    seg, _, _ = shortest_path(Cl, P[i], P[j], margin=6)
    if not seg: return P
    Q = remove_loops(P[:i] + seg + P[j + 1:])
    sc = lambda R: lam * f_eco(R, E) + (1 - lam) * f_len(R) / Lref * Ebar
    return Q if sc(Q) < sc(P) else P

def random_path(shape, s, g, rng):
    ones = np.ones(shape); pts = [s]
    for _ in range(int(rng.integers(1, 4))):
        pts.append((int(rng.integers(0, shape[0])), int(rng.integers(0, shape[1]))))
    pts.append(g); P = [s]
    for u, v in zip(pts[:-1], pts[1:]):
        seg, _, _ = shortest_path(ones, u, v); P += seg[1:]
    return remove_loops(P)

def _nds(F):
    n = len(F); S = [[] for _ in range(n)]; cnt = np.zeros(n, int); fronts = [[]]
    for p in range(n):
        for q in range(n):
            if p == q: continue
            if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]): S[p].append(q)
            elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]): cnt[p] += 1
        if cnt[p] == 0: fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                cnt[q] -= 1
                if cnt[q] == 0: nxt.append(q)
        i += 1; fronts.append(nxt)
    return fronts[:-1]

def _crowd(F, idx):
    d = np.zeros(len(idx)); sub = F[idx]
    for m in range(F.shape[1]):
        o = np.argsort(sub[:, m]); d[o[0]] = d[o[-1]] = np.inf
        rng_ = sub[o[-1], m] - sub[o[0], m] or 1.0
        for k in range(1, len(idx) - 1): d[o[k]] += (sub[o[k + 1], m] - sub[o[k - 1], m]) / rng_
    return d

def nsga2(C, E, s, g, pop=40, gens=45, pc=0.9, pm=0.5, memetic=False, lam=0.5, seed=0, record=False):
    rng = np.random.default_rng(seed); shape = C.shape
    seedP, _, _ = shortest_path(C, s, g)
    Lref = f_len(seedP); Ebar = float(E.mean())
    if memetic:
        P = [seedP] + [mutate(seedP, C, rng) for _ in range(pop // 2 - 1)] + [random_path(shape, s, g, rng) for _ in range(pop - pop // 2)]
    else:
        P = [random_path(shape, s, g, rng) for _ in range(pop)]
    obj = lambda R: (f_eco(R, E), f_len(R))
    F = np.array([obj(R) for R in P]); hist = []
    for gen in range(gens):
        fronts = _nds(F); rank = np.empty(len(P), int); cd = np.empty(len(P))
        for fi, fr in enumerate(fronts): rank[fr] = fi; cd[fr] = _crowd(F, fr)
        def tour():
            a, b = rng.integers(len(P), size=2)
            return a if (rank[a] < rank[b] or (rank[a] == rank[b] and cd[a] > cd[b])) else b
        kids = []
        while len(kids) < pop:
            p1, p2 = P[tour()], P[tour()]
            c1, c2 = crossover(p1, p2, rng) if rng.random() < pc else (list(p1), list(p2))
            for c in (c1, c2):
                if rng.random() < pm: c = mutate(c, C, rng)
                if memetic: c = local_search(c, E, Ebar, lam, Lref, rng)
                kids.append(c)
        R = P + kids[:pop]; FR = np.vstack([F, [obj(x) for x in kids[:pop]]])
        fronts = _nds(FR); newi = []
        for fr in fronts:
            if len(newi) + len(fr) <= pop: newi += fr
            else:
                d = _crowd(FR, fr); newi += [fr[k] for k in np.argsort(-d)[:pop - len(newi)]]; break
        P = [R[i] for i in newi]; F = FR[newi]
        if record: hist.append(F[_nds(F)[0]].copy())
    f0 = _nds(F)[0]
    _, u = np.unique(np.round(F[f0], 6), axis=0, return_index=True)
    front = [P[f0[i]] for i in sorted(u)]; FF = np.array([obj(x) for x in front])
    nrm = (FF - FF.min(0)) / np.where(np.ptp(FF, 0) == 0, 1, np.ptp(FF, 0))
    knee = front[int(np.argmin(np.hypot(nrm[:, 0], nrm[:, 1])))]
    mini = front[int(np.argmin(FF[:, 0]))]
    pacc = front[int(np.argmin(FF[:, 0] * FF[:, 1]))]
    return dict(front=front, F=FF, knee=knee, min_eco=mini, p_acc=pacc, hist=hist, seed_route=seedP)
