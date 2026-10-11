"""Turn results/*.pkl into results/summary.json and results/figures/*.png (all paper tables and figures)."""
import sys, os, json, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
from vegsens.scenes import make_scene
from vegsens.indices import indices
from vegsens.metrics import hypervolume_2d, f_len, f_eco, a_eco
from vegsens import CLASSES
R = os.path.join(os.path.dirname(__file__), "..", "results"); FD = os.path.join(R, "figures"); os.makedirs(FD, exist_ok=True)
L = lambda n: pickle.load(open(os.path.join(R, n), "rb"))
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9, "axes.grid": True, "grid.alpha": .3})
SOLV = ["A*", "Dijkstra", "ACO", "NSGA-II", "Memetic"]; SC = ["S1", "S2", "S3", "S4", "S5"]; REP = "S3"
COL = {"A*": "#1f77b4", "Dijkstra": "#7f7f7f", "ACO": "#2ca02c", "NSGA-II": "#ff7f0e", "Memetic": "#d62728"}
S = {}
def sv(fig, n): fig.savefig(os.path.join(FD, n), dpi=200, bbox_inches="tight"); plt.close(fig)
CMAP = matplotlib.colors.ListedColormap(["#2b6cb0", "#a0a0a0", "#c8a46e", "#9acd32", "#1e5b1e"])

# ---------------- segmentation ----------------
seg = L("seg.pkl"); sd = pd.DataFrame(seg["rows"])
rep = sd[sd.scene == REP]
S["seg_rep"] = rep[["variant", "chi", "OA", "kappa", "classes"]].round(3).to_dict("records")
S["seg_all"] = sd[["scene", "variant", "chi", "OA", "kappa", "classes"]].round(3).to_dict("records")
full = rep[rep.variant.str.startswith("Full (")].iloc[0]; km = rep[rep.variant == "K-means baseline"].iloc[0]
S["seg_full_recall"] = dict(zip(CLASSES, np.round(full.recall, 3)))
fs = sd[sd.variant.str.startswith("Full (")]; S["seg_full_range"] = dict(OA=[fs.OA.min(), fs.OA.max()], kappa=[fs.kappa.min(), fs.kappa.max()], chi=[fs.chi.min(), fs.chi.max()])
sw = pd.DataFrame(seg["sweep"]); S["slic_sweep"] = sw.groupby("div")[["chi", "OA", "kappa"]].agg(["mean", "min"]).round(3).reset_index().values.tolist()
scenes = {s: make_scene(s) for s in SC}
S["landcover_ref"] = {s: (np.bincount(scenes[s]["label"].ravel(), minlength=5) / scenes[s]["label"].size * 100).round(2).tolist() for s in SC}
fullmap = seg["maps"]["Full (SLIC + GMM + MRF-lite)"]
S["landcover_map"] = (np.bincount(fullmap.ravel(), minlength=5) / fullmap.size * 100).round(2).tolist()

def rgb(img): x = img[..., :3]; x = x / np.percentile(x, 99.5); return np.clip(x, 0, 1)
fig, ax = plt.subplots(2, 5, figsize=(12, 5.2))
for i, s in enumerate(SC):
    sc = scenes[s]; ax[0, i].imshow(rgb(sc["img"])); ax[1, i].imshow(sc["label"], cmap=CMAP, vmin=0, vmax=4)
    for a in ax[:, i]:
        a.plot(sc["start"][1], sc["start"][0], "wo", ms=6, mec="k"); a.plot(sc["goal"][1], sc["goal"][0], "ws", ms=6, mec="k"); a.set_xticks([]); a.set_yticks([]); a.grid(False)
    ax[0, i].set_title(f"{s} (dense {S['landcover_ref'][s][4]:.0f}%)")
ax[0, 0].set_ylabel("RGB rendering"); ax[1, 0].set_ylabel("Construction labels")
h = [plt.Rectangle((0, 0), 1, 1, color=CMAP(i)) for i in range(5)]; fig.legend(h, CLASSES, loc="lower center", ncol=5, frameon=False); sv(fig, "fig_samples.png")
sc3 = scenes[REP]; idx3 = indices(sc3["img"])
fig, a = plt.subplots(figsize=(5.5, 5)); a.imshow(rgb(sc3["img"])); a.plot(sc3["start"][1], sc3["start"][0], "wo", ms=8, mec="k"); a.plot(sc3["goal"][1], sc3["goal"][0], "ws", ms=8, mec="k"); a.set_xticks([]); a.set_yticks([]); a.grid(False); sv(fig, "fig_scene.png")
fig, ax = plt.subplots(1, 3, figsize=(12, 4.2))
for a, (t, m) in zip(ax, [("(a) Construction reference", sc3["label"]), (f"(b) K-means (χ={km.chi:.2f}, OA={km.OA:.2f}, {km.classes} classes)", seg["maps"]["K-means baseline"]), (f"(c) SLIC+GMM+MRF-lite (χ={full.chi:.2f}, OA={full.OA:.2f})", fullmap)]):
    a.imshow(m, cmap=CMAP, vmin=0, vmax=4); a.set_title(t, fontsize=9); a.set_xticks([]); a.set_yticks([]); a.grid(False)
fig.legend(h, CLASSES, loc="lower center", ncol=5, frameon=False); sv(fig, "fig_seg.png")
for key, nm in [("NDVI", "fig_ndvi.png")]:
    fig, a = plt.subplots(figsize=(5.8, 4.6)); im = a.imshow(idx3[key], cmap="RdYlGn"); plt.colorbar(im, ax=a, label=key); a.set_xticks([]); a.set_yticks([]); a.grid(False); sv(fig, nm)
sens = L("sens.pkl"); E3 = sens["E"]
S["class_ess"] = [float(E3[fullmap == c].mean()) if np.any(fullmap == c) else float("nan") for c in range(5)]
S["class_ess_ref"] = [float(E3[sc3["label"] == c].mean()) for c in range(5)]
fig, a = plt.subplots(figsize=(5.8, 4.6)); im = a.imshow(E3, cmap="magma"); plt.colorbar(im, ax=a, label="ESS"); a.set_xticks([]); a.set_yticks([]); a.grid(False); sv(fig, "fig_ess.png")
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for a, k in zip(ax, ["ExG", "GLI"]):
    im = a.imshow(idx3[k], cmap="YlGn"); plt.colorbar(im, ax=a, label=k); a.set_title(k); a.set_xticks([]); a.set_yticks([]); a.grid(False)
sv(fig, "fig_exggli.png")
S["gli_range"] = [float(idx3["GLI"].min()), float(idx3["GLI"].max())]
cm = np.array(full.cm, float); cmn = cm / cm.sum(1, keepdims=True)
fig, a = plt.subplots(figsize=(5.4, 4.6)); im = a.imshow(cmn, cmap="Greens", vmin=0, vmax=1)
for i in range(5):
    for j in range(5): a.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", color="w" if cmn[i, j] > .5 else "k", fontsize=8)
a.set_xticks(range(5)); a.set_xticklabels(CLASSES, rotation=30); a.set_yticks(range(5)); a.set_yticklabels(CLASSES); a.set_xlabel("Predicted"); a.set_ylabel("Construction reference"); a.grid(False); plt.colorbar(im, ax=a); sv(fig, "fig_conf.png")
fig, a = plt.subplots(figsize=(5.5, 3.4)); x = np.arange(5)
a.bar(x - .2, S["landcover_ref"][REP], .4, label="Construction labels", color="#bbbbbb"); a.bar(x + .2, S["landcover_map"], .4, label="Segmented map", color=[CMAP(i) for i in range(5)])
a.set_xticks(x); a.set_xticklabels(CLASSES); a.set_ylabel("Area (%)"); a.legend(frameon=False); sv(fig, "fig_lc.png")
fig, a = plt.subplots(figsize=(6.4, 3.4)); v = rep.set_index("variant")
names = list(v.index); xx = np.arange(len(names))
a.bar(xx - .2, v.chi, .4, label="χ"); a.bar(xx + .2, v.OA, .4, label="OA")
a.set_xticks(xx); a.set_xticklabels([n.replace(" (", "\n(").replace(", k", ",\nk") for n in names], fontsize=6.5); a.set_ylim(0, 1.05); a.legend(frameon=False); sv(fig, "fig_segabl.png")
fig, a = plt.subplots(figsize=(5.4, 3.4)); g = sw.groupby("div").mean(numeric_only=True)
a.plot(220 * 220 / g.index, g.chi, "o-", label="χ"); a.plot(220 * 220 / g.index, g.OA, "s-", label="OA"); a.plot(220 * 220 / g.index, g.kappa, "^-", label="κ")
a.set_xscale("log"); a.set_xlabel("SLIC target segments (log)"); a.set_ylabel("Mean over S1–S5"); a.axvline(220 * 220 / 120, ls="--", c="k", lw=.8); a.legend(frameon=False); sv(fig, "fig_slic.png")

# ---------------- sensitivity ----------------
corr = np.array(sens["corr"]); S["corr"] = np.round(corr, 3).tolist(); S["vif"] = np.round(sens["vif"], 2).tolist()
dr = pd.DataFrame(sens["dirichlet"]); S["dirichlet"] = dict(jac_median=float(dr.jac.median()), jac_q=[float(dr.jac.quantile(.25)), float(dr.jac.quantile(.75))],
     f_iqr=float(dr.f_eco.quantile(.75) - dr.f_eco.quantile(.25)), a_iqr=float(dr.a_eco.quantile(.75) - dr.a_eco.quantile(.25)),
     f_median=float(dr.f_eco.median()), a_median=float(dr.a_eco.median()))
S["w1"] = sens["w1"]; S["alpha"] = sens["alpha"]; S["layers"] = sens["layers"]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4)); w1 = pd.DataFrame(sens["w1"])
ax[0].plot(w1.w1, w1.jac, "o-", label="Jaccard overlap with default route"); ax[0].set_xlabel("NDVI weight w1"); ax[0].set_ylim(0, 1.05); a2 = ax[0].twinx(); a2.plot(w1.w1, w1.a_eco, "s--", c="C3", label="A_eco"); a2.set_ylabel("A_eco (px·ESS)"); ax[0].legend(loc="lower left", frameon=False, fontsize=7)
ax[1].hist(dr.jac, bins=25, color="#888"); ax[1].set_xlabel("Jaccard overlap with default route"); ax[1].set_ylabel("Count (300 Dirichlet draws)"); sv(fig, "fig_weights.png")
al = pd.DataFrame(sens["alpha"]); fig, a = plt.subplots(figsize=(5.4, 3.4))
a.plot(al.alpha, al.f_eco, "o-", label="mean ESS"); a.set_xlabel("Eco-cost weight α"); a.set_ylabel("Mean ESS"); a2 = a.twinx(); a2.plot(al.alpha, al.length, "s--", c="C3", label="length"); a2.set_ylabel("Length (px)")
a.axvline(1.2, ls=":", c="k"); a.legend(loc="upper center", frameon=False); a2.legend(loc="center right", frameon=False); sv(fig, "fig_alpha.png")
pg = pd.DataFrame(sens["popgen"])

# ---------------- routing ----------------
rt = L("route.pkl"); lg = pd.DataFrame(rt["log"]); fr = rt["fronts"]
main = lg[(lg.kind.isin(["route", "knee"]))].copy()
med = main.groupby(["scene", "solver"])[["length", "f_eco", "a_eco", "ndvi", "dense", "time"]].median()
S["route_rep"] = {s: med.loc[(REP, s)].round(4).to_dict() for s in SOLV}
S["route_rep_iqr"] = {s: main[(main.scene == REP) & (main.solver == s)][["f_eco", "a_eco", "length"]].quantile([.25, .75]).round(4).values.tolist() for s in ["ACO", "NSGA-II", "Memetic"]}
for kind in ["min_eco", "p_acc"]:
    S["mem_" + kind] = lg[(lg.scene == REP) & (lg.solver == "Memetic") & (lg.kind == kind)][["length", "f_eco", "a_eco", "ndvi", "dense"]].median().round(4).to_dict()
    S["nsga_" + kind] = lg[(lg.scene == REP) & (lg.solver == "NSGA-II") & (lg.kind == kind)][["length", "f_eco", "a_eco", "ndvi", "dense"]].median().round(4).to_dict()
astar_J = lg[(lg.solver == "A*")].set_index("scene").J; dij_J = lg[(lg.solver == "Dijkstra")].set_index("scene").J
S["astar_vs_dij"] = dict(maxJdiff=float((astar_J - dij_J).abs().max()), expanded_A=lg[lg.solver == "A*"].expanded.tolist(), expanded_D=lg[lg.solver == "Dijkstra"].expanded.tolist())
S["multi"] = {m: {s: [float(med.loc[(sc, s)][m]) for sc in SC] for s in SOLV} for m in ["f_eco", "a_eco", "length", "dense"]}
S["multi_summary"] = {m: {s: [float(np.mean(S["multi"][m][s])), float(np.std(S["multi"][m][s], ddof=1))] for s in SOLV} for m in ["f_eco", "a_eco"]}
cv = main[main.solver.isin(["ACO", "NSGA-II", "Memetic"])].groupby(["scene", "solver"]).f_eco.agg(lambda x: x.std(ddof=1) / x.mean())
S["cv_runs"] = {s: [float(cv.loc[(sc, s)]) for sc in SC] for s in ["ACO", "NSGA-II", "Memetic"]}
S["n_runs"] = int(main[main.solver == "Memetic"].run.nunique())
# hypervolume
hvs = {}; hvhist = {}
for scn in SC:
    pts = []
    for (s, nm, r), d in fr.items():
        if s != scn: continue
        if "F" in d: pts.append(d["F"])
    sco = scenes[scn]
    for nm in ["A*", "Dijkstra"]:
        P = fr[(scn, nm, 0)]["route"]
    U = np.vstack(pts); lo, hi = U.min(0), U.max(0)
    norm = lambda F: (np.asarray(F) - lo) / np.where(hi - lo == 0, 1, hi - lo)
    for (s, nm, r), d in fr.items():
        if s == scn and "F" in d:
            hvs.setdefault((scn, nm), []).append(hypervolume_2d(norm(d["F"])))
            if scn == REP: hvhist.setdefault(nm, []).append([hypervolume_2d(norm(F)) for F in d["hist"]])
S["hv"] = {f"{s}|{nm}": [float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75))] for (s, nm), v in hvs.items()}
S["hv_rep"] = {nm: S["hv"][f"{REP}|{nm}"] for nm in ["NSGA-II", "Memetic"]}
fig, a = plt.subplots(figsize=(5.6, 3.4))
for nm in ["NSGA-II", "Memetic"]:
    H = np.array(hvhist[nm]); g = np.arange(1, H.shape[1] + 1); a.plot(g, np.median(H, 0), c=COL[nm], label=nm); a.fill_between(g, np.percentile(H, 25, 0), np.percentile(H, 75, 0), color=COL[nm], alpha=.2)
a.set_xlabel("Generation"); a.set_ylabel("Hypervolume (normalised)"); a.legend(frameon=False); sv(fig, "fig_hv.png")
S["hv_gen10"] = {nm: float(np.median(np.array(hvhist[nm])[:, 9])) for nm in ["NSGA-II", "Memetic"]}
# routes figure
ndvi3 = idx3["NDVI"]; fig, a = plt.subplots(figsize=(6.2, 5.2)); a.imshow(ndvi3, cmap="Greens", alpha=.85)
for nm, P in [("A*", fr[(REP, "A*", 0)]["route"]), ("Dijkstra", fr[(REP, "Dijkstra", 0)]["route"]), ("ACO", fr[(REP, "ACO", 0)]["route"]), ("NSGA-II", fr[(REP, "NSGA-II", 0)]["knee"]), ("Memetic", fr[(REP, "Memetic", 0)]["knee"])]:
    p = np.asarray(P); a.plot(p[:, 1], p[:, 0], c=COL[nm], lw=1.4 if nm != "Dijkstra" else 2.6, ls="-" if nm != "Dijkstra" else ":", label=nm)
a.plot(sc3["start"][1], sc3["start"][0], "wo", ms=8, mec="k"); a.plot(sc3["goal"][1], sc3["goal"][0], "ws", ms=8, mec="k"); a.legend(fontsize=7, loc="lower left"); a.set_xticks([]); a.set_yticks([]); a.grid(False); sv(fig, "fig_routes.png")
# pareto
fig, a = plt.subplots(figsize=(5.8, 3.8))
for nm in ["NSGA-II", "Memetic"]:
    F = fr[(REP, nm, 0)]["F"]; o = np.argsort(F[:, 1]); a.plot(F[o, 1], F[o, 0], "o-", ms=3, c=COL[nm], label=f"{nm} front (run 0)")
d = fr[(REP, "Memetic", 0)]
for P, mk, lb in [(d["knee"], "*", "knee"), (d["min_eco"], "v", "min $f_{eco}$"), (d["p_acc"], "D", "$P_{acc}$ (min $f_{len}f_{eco}$)")]:
    a.plot(f_len(P), f_eco(P, E3), mk, ms=10, c="k", mfc="none", label=lb)
PA = fr[(REP, "A*", 0)]["route"]; a.plot(f_len(PA), f_eco(PA, E3), "s", c=COL["A*"], ms=8, label="A*")
a.set_xlabel("Route length $f_{len}$ (px)"); a.set_ylabel("Mean ESS $f_{eco}$"); a.legend(fontsize=7, frameon=False); sv(fig, "fig_pareto.png")
# bars
fig, ax = plt.subplots(1, 4, figsize=(12, 3))
for a, (m, t) in zip(ax, [("f_eco", "Mean ESS"), ("length", "Length (px)"), ("ndvi", "Mean NDVI exposure"), ("dense", "Dense-vegetation cells")]):
    vals = [S["route_rep"][s][m] for s in SOLV]
    err = [[0, 0]] * 2 + [[S["route_rep"][s][m] - main[(main.scene == REP) & (main.solver == s)][m].quantile(.25), main[(main.scene == REP) & (main.solver == s)][m].quantile(.75) - S["route_rep"][s][m]] for s in ["ACO", "NSGA-II", "Memetic"]]
    a.bar(range(5), vals, yerr=np.array(err).T, color=[COL[s] for s in SOLV], capsize=3); a.set_xticks(range(5)); a.set_xticklabels(SOLV, rotation=30, fontsize=7); a.set_title(t, fontsize=9)
sv(fig, "fig_bars.png")
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
for a, (m, t) in zip(ax, [("f_eco", "(a) Mean ESS $f_{eco}$"), ("a_eco", "(b) Route-accumulated ESS $A_{eco}$ (px·ESS)")]):
    v = [S["route_rep"][s][m] for s in SOLV]; a.bar(range(5), v, color=[COL[s] for s in SOLV])
    for i, x in enumerate(v): a.text(i, x, f"{x:.3f}" if m == "f_eco" else f"{x:.1f}", ha="center", va="bottom", fontsize=7)
    a.set_xticks(range(5)); a.set_xticklabels(SOLV, fontsize=8); a.set_title(t, fontsize=9)
sv(fig, "fig_acc.png")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
for a, m, t in [(ax[0], "f_eco", "Mean ESS (per-scene median)"), (ax[1], "a_eco", "Route-accumulated ESS (per-scene median)")]:
    for s in SOLV: a.plot(SC, S["multi"][m][s], marker="o", c=COL[s], ls=":" if s == "Dijkstra" else "-", lw=2.5 if s == "Dijkstra" else 1.3, label=s)
    a.set_title(t, fontsize=9); a.set_xlabel("Scene (increasing vegetation density)")
ax[0].legend(fontsize=7, frameon=False); sv(fig, "fig_multi.png")
fig, ax = plt.subplots(1, 5, figsize=(13, 3), sharey=True)
for a, scn in zip(ax, SC):
    data = [main[(main.scene == scn) & (main.solver == s)].f_eco.values for s in ["ACO", "NSGA-II", "Memetic"]]
    bp = a.boxplot(data, patch_artist=True, widths=.6)
    for p, s in zip(bp["boxes"], ["ACO", "NSGA-II", "Memetic"]): p.set_facecolor(COL[s]); p.set_alpha(.6)
    a.axhline(med.loc[(scn, "A*")].f_eco, c=COL["A*"], ls="--", lw=1, label="A* = Dijkstra")
    a.set_xticks([1, 2, 3]); a.set_xticklabels(["ACO", "NSGA-II", "Memetic"], fontsize=7, rotation=20); a.set_title(scn)
ax[0].set_ylabel(f"Mean ESS ({S['n_runs']} runs)"); ax[0].legend(fontsize=7, frameon=False); sv(fig, "fig_box.png")
# detour
Lmin = med.loc[(REP, "A*")].length; fig, a = plt.subplots(figsize=(5.4, 3.6))
for s in SOLV:
    sub = main[(main.scene == REP) & (main.solver == s)]
    a.scatter(sub.length / Lmin, sub.f_eco, c=COL[s], s=12 if s in ["ACO", "NSGA-II", "Memetic"] else 60, alpha=.6, label=s, marker="o" if s != "Dijkstra" else "x")
a.set_xlabel("Detour ratio $D_r$ (length / A* length)"); a.set_ylabel("Mean ESS"); a.legend(fontsize=7, frameon=False); sv(fig, "fig_detour.png")

# ---------------- statistics ----------------
def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); out = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, (m - r) * p[i]); out[i] = min(1, run)
    return out
st = {}
for m in ["f_eco", "a_eco"]:
    rows = []
    mem = np.array(S["multi"][m]["Memetic"])
    for b in ["A*", "Dijkstra", "ACO", "NSGA-II"]:
        d = np.array(S["multi"][m][b]) - mem
        t, p = stats.ttest_rel(np.array(S["multi"][m][b]), mem)
        try: w = stats.wilcoxon(d, method="exact").pvalue
        except Exception: w = float("nan")
        rows.append(dict(baseline=b, mean=float(d.mean()), sd=float(d.std(ddof=1)), dz=float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float("nan"), t=float(t), p=float(p), wilcoxon=float(w)))
    for r, h in zip(rows, holm([r["p"] for r in rows])): r["p_holm"] = float(h)
    st[m] = rows
S["stats"] = st
pw = {}
for m in ["f_eco", "a_eco"]:
    for scn in SC:
        mv = main[(main.scene == scn) & (main.solver == "Memetic")][m].values
        for b in ["A*", "Dijkstra"]:
            ref = float(med.loc[(scn, b)][m]); dd = mv - ref
            pw[(m, scn, b)] = float(stats.wilcoxon(dd).pvalue) if np.any(dd != 0) else 1.0
        for b in ["ACO", "NSGA-II"]:
            pw[(m, scn, b)] = float(stats.mannwhitneyu(mv, main[(main.scene == scn) & (main.solver == b)][m].values).pvalue)
S["within"] = {f"{m}|{s}|{b}": v for (m, s, b), v in pw.items()}
S["within_dir"] = {f"{m}|{s}|{b}": float(np.median(main[(main.scene == s) & (main.solver == "Memetic")][m]) - med.loc[(s, b)][m]) for (m, s, b) in pw}
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
for a, m, t in [(ax[0], "f_eco", "Mean ESS"), (ax[1], "a_eco", "Route-accumulated ESS")]:
    M = np.array([[pw[(m, s, b)] for s in SC] for b in ["A*", "Dijkstra", "ACO", "NSGA-II"]])
    D = np.array([[S["within_dir"][f"{m}|{s}|{b}"] for s in SC] for b in ["A*", "Dijkstra", "ACO", "NSGA-II"]])
    im = a.imshow(-np.log10(np.maximum(M, 1e-12)), cmap="viridis", vmin=0, vmax=12)
    for i in range(4):
        for j in range(5): a.text(j, i, ("−" if D[i, j] < 0 else "+") + f"\n{M[i, j]:.0e}", ha="center", va="center", fontsize=6.5, color="w")
    a.set_xticks(range(5)); a.set_xticklabels(SC); a.set_yticks(range(4)); a.set_yticklabels(["vs A*", "vs Dijkstra", "vs ACO", "vs NSGA-II"]); a.set_title(t, fontsize=9); a.grid(False)
plt.colorbar(im, ax=ax, label="−log10 p"); sv(fig, "fig_pval.png")
fig, a = plt.subplots(figsize=(6, 3.2)); x = np.arange(4)
for k, (m, lb) in enumerate([("f_eco", "mean ESS"), ("a_eco", "accumulated ESS")]):
    dz = [r["dz"] for r in st[m]]; a.bar(x + (k - .5) * .38, np.clip(dz, -20, 20), .38, label=lb)
a.axhline(0, c="k", lw=.8); a.set_xticks(x); a.set_xticklabels(["vs A*", "vs Dijkstra", "vs ACO", "vs NSGA-II"]); a.set_ylabel("$d_z$ (positive favours memetic; clipped ±20)"); a.legend(frameon=False); sv(fig, "fig_dz.png")

# ---------------- efficiency ----------------
ef = L("eff.pkl"); mem_ = pd.DataFrame(ef["mem"]).groupby("solver").peak_mb.median(); sc_ = pd.DataFrame(ef["scal"])
S["eff"] = {s: dict(time=float(main[(main.scene == REP) & (main.solver == s)].time.median()), mem=float(mem_[s])) for s in SOLV}
S["scal"] = sc_.round(4).to_dict("records")
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.2))
ax[0].bar(range(5), [S["eff"][s]["time"] for s in SOLV], color=[COL[s] for s in SOLV]); ax[0].set_yscale("log"); ax[0].set_ylabel("Runtime per scene (s, log)")
ax[1].bar(range(5), [S["eff"][s]["mem"] for s in SOLV], color=[COL[s] for s in SOLV]); ax[1].set_ylabel("Peak traced memory (MB)")
for a in ax: a.set_xticks(range(5)); a.set_xticklabels(SOLV, fontsize=8)
sv(fig, "fig_eff.png")
fig, a = plt.subplots(figsize=(5.4, 3.4))
for s in SOLV:
    d = sc_[sc_.solver == s]; a.plot(d.cells, d.time, "o-", c=COL[s], label=s)
a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("Grid cells"); a.set_ylabel("Runtime (s)"); a.legend(fontsize=7, frameon=False); sv(fig, "fig_scal.png")
pgs = pg.copy()
pts = np.vstack([fr[k]["F"] for k in fr if k[0] == REP and "F" in fr[k]] + list(pgs.F)); lo, hi = pts.min(0), pts.max(0)
pgs["hv"] = [hypervolume_2d((np.asarray(F) - lo) / (hi - lo)) for F in pgs.F]
S["popgen"] = pgs.groupby(["pop", "gens"])[["hv", "min_eco", "knee_a"]].median().round(4).reset_index().values.tolist()
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.2))
a1 = pgs[pgs.gens == 45].groupby("pop").hv.median(); a2 = pgs[pgs["pop"] == 40].groupby("gens").hv.median()
ax[0].plot(a1.index, a1.values, "o-"); ax[0].set_xlabel("Population size (45 generations)"); ax[0].set_ylabel("Median hypervolume (5 runs)")
ax[1].plot(a2.index, a2.values, "s-", c="C3"); ax[1].set_xlabel("Generations (population 40)")
sv(fig, "fig_popgen.png")
# summary heatmap
mets = ["length", "f_eco", "a_eco", "ndvi", "dense"]; raw = np.array([[S["route_rep"][s][m] for m in mets] + [S["eff"][s]["time"]] for s in SOLV])
nm_ = (raw - raw.min(0)) / np.where(np.ptp(raw, 0) == 0, 1, np.ptp(raw, 0))
fig, a = plt.subplots(figsize=(7.5, 3.3)); im = a.imshow(nm_, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
for i in range(5):
    for j in range(6): a.text(j, i, f"{raw[i, j]:.3f}" if j in (1, 3) else (f"{raw[i, j]:.2f}" if j == 5 else f"{raw[i, j]:.1f}"), ha="center", va="center", fontsize=7.5)
a.set_xticks(range(6)); a.set_xticklabels(["Length", "Mean ESS", "Accum. ESS", "NDVI exp.", "Dense cells", "Runtime (s)"], fontsize=8); a.set_yticks(range(5)); a.set_yticklabels(SOLV); a.grid(False)
plt.colorbar(im, ax=a, label="column-wise rank (0 = best)"); sv(fig, "fig_summary.png")
json.dump(S, open(os.path.join(R, "summary.json"), "w"), indent=1, default=float)
print("analysis done")
