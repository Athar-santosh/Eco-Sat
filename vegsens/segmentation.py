"""Label-free superpixel-mixture segmentation (SLIC + GMM + rank rules + MRF-lite) and baselines."""
import numpy as np
from skimage.segmentation import slic
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import uniform_filter
from . import WATER, URBAN, BARREN, SPARSE, DENSE

MRF_WINDOW, MRF_ITERS = 3, 2
SLIC_COMPACTNESS = 0.10

def features(img, idx):
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    return np.stack([R, G, B, idx["NDVI"], idx["VARI"], idx["GLI"]], -1)

def _split_gap(vals):
    """indices sorted desc by vals, split at the largest consecutive gap -> (upper, lower)."""
    o = np.argsort(-vals); g = -np.diff(vals[o]); k = int(np.argmax(g)) + 1
    return list(o[:k]), list(o[k:])

def assign_classes(means, rule="gap"):
    """Algorithm 4. means: (k,6) component means in original units [R,G,B,NDVI,VARI,GLI].
    rule='gap' (default, threshold-free, any k>=3) or 'rank5' (earlier two-highest-NDVI rule, k=5)."""
    k = len(means); n = means[:, 3]; b = means[:, :3].mean(1)
    beta = means[:, 2] / (means[:, :3].sum(1) + 1e-9); R, G = means[:, 0], means[:, 1]
    if rule == "rank5":
        order = np.argsort(-n); m = {order[0]: DENSE, order[1]: SPARSE}
        rest = list(order[2:]); water = max(rest, key=lambda c: beta[c] - b[c]); m[water] = WATER
        rest = [c for c in rest if c != water]
        hi, lo = (rest[0], rest[1]) if b[rest[0]] >= b[rest[1]] else (rest[1], rest[0])
        m[hi] = BARREN; m[lo] = URBAN; return m
    # step 1: three NDVI groups separated by the two largest gaps in sorted component NDVI
    o = np.argsort(-n); gaps = -np.diff(n[o]); cut = sorted(np.argsort(-gaps)[:2] + 1)
    dense, sparse, nonveg = list(o[:cut[0]]), list(o[cut[0]:cut[1]]), list(o[cut[1]:])
    m = {c: DENSE for c in dense}; m.update({c: SPARSE for c in sparse})
    # step 2: non-vegetation components
    def soil_or_built(c): return BARREN if R[c] > G[c] else URBAN
    if len(nonveg) == 1:
        c = nonveg[0]; m[c] = WATER if beta[c] > 1 / 3 and b[c] < b.mean() else soil_or_built(c)
        return m
    ws = beta[nonveg] - b[nonveg]; up, lo = _split_gap(ws)
    for i in up: m[nonveg[i]] = WATER
    lo = [nonveg[i] for i in lo]
    if len(lo) == 1: m[lo[0]] = soil_or_built(lo[0])
    elif lo:
        bu, bl = _split_gap(b[lo])
        for i in bu: m[lo[i]] = BARREN
        for i in bl: m[lo[i]] = URBAN
    return m

def mrf_lite(lab, window=MRF_WINDOW, iters=MRF_ITERS, k=5):
    lab = lab.copy()
    for _ in range(iters):
        counts = np.stack([uniform_filter((lab == c).astype(float), window, mode="nearest") for c in range(k)])
        best = counts.argmax(0)
        cur = np.take_along_axis(counts, lab[None].astype(int), 0)[0]
        lab = np.where(counts.max(0) > cur + 1e-12, best, lab).astype(np.int8)
    return lab

def segment(img, idx, method="full", n_segments=None, seed=0, k=8, rule="gap"):
    H, W = img.shape[:2]
    X = features(img, idx)
    if method in ("full", "no_mrf"):
        ns = n_segments or int(H * W / 120)
        sp = slic(img, n_segments=ns, compactness=SLIC_COMPACTNESS, start_label=0, channel_axis=-1, convert2lab=False)  # 4 raw bands
        nsp = sp.max() + 1
        F = np.stack([np.bincount(sp.ravel(), X[..., j].ravel(), nsp) / np.bincount(sp.ravel(), minlength=nsp) for j in range(6)], 1)
        Z = StandardScaler().fit_transform(F)
        g = GaussianMixture(k, covariance_type="full", random_state=seed, n_init=3).fit(Z)
        comp = g.predict(Z)
        means = np.stack([F[comp == c].mean(0) if np.any(comp == c) else F.mean(0) for c in range(k)])
        m = assign_classes(means, rule)
        lab = np.vectorize(m.get)(comp)[sp].astype(np.int8)
        return mrf_lite(lab) if method == "full" else lab
    Xf = X.reshape(-1, 6); Z = StandardScaler().fit_transform(Xf)
    if method == "pixel_gmm":
        rng = np.random.default_rng(seed); sub = rng.choice(len(Z), 8000, replace=False)
        comp = GaussianMixture(k, covariance_type="full", random_state=seed, n_init=3).fit(Z[sub]).predict(Z)
    else:  # kmeans, kmeans_mrf
        comp = KMeans(k, n_init=5, random_state=seed).fit_predict(Z)
    means = np.stack([Xf[comp == c].mean(0) for c in range(k)])
    m = assign_classes(means, rule)
    lab = np.vectorize(m.get)(comp).reshape(H, W).astype(np.int8)
    return mrf_lite(lab) if method == "kmeans_mrf" else lab

def coherence(lab):
    eq = [(lab[:, 1:] == lab[:, :-1]), (lab[1:] == lab[:-1]), (lab[1:, 1:] == lab[:-1, :-1]), (lab[1:, :-1] == lab[:-1, 1:])]
    return float(sum(e.sum() for e in eq) / sum(e.size for e in eq))

def accuracy(pred, ref, k=5):
    cm = np.zeros((k, k), int)
    np.add.at(cm, (ref.ravel(), pred.ravel()), 1)
    N = cm.sum(); oa = np.trace(cm) / N
    pe = (cm.sum(1) * cm.sum(0)).sum() / N ** 2
    kappa = (oa - pe) / (1 - pe)
    recall = cm.diagonal() / np.maximum(cm.sum(1), 1)
    return dict(OA=float(oa), kappa=float(kappa), recall=recall.tolist(), cm=cm.tolist(),
                classes_present=int(len(np.unique(pred))))
