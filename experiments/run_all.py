"""Run every experiment reported in the paper and write results/ (pickles + CSV logs).
Usage: python experiments/run_all.py [--runs 40] [--stage all|seg|route|sens|eff]"""
import sys, os, time, pickle, argparse, tracemalloc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from vegsens.scenes import make_scene
from vegsens.indices import indices, ess, cost_surface, slope, minmax, DEFAULT_W
from vegsens.segmentation import segment, coherence, accuracy
from vegsens.solvers import astar, dijkstra, aco, nsga2
from vegsens.metrics import route_metrics, f_eco, a_eco
import vegsens.segmentation as SEG

OUT = os.path.join(os.path.dirname(__file__), "..", "results"); os.makedirs(OUT, exist_ok=True)
SCENES = ["S1", "S2", "S3", "S4", "S5"]; REP = "S3"
def save(obj, name): pickle.dump(obj, open(os.path.join(OUT, name), "wb"))

def prep(name, size=220):
    sc = make_scene(name, size=size) if size == 220 else make_scene(name, size=size)
    idx = indices(sc["img"]); E = ess(idx); lab = segment(sc["img"], idx, "full")
    return sc, idx, E, lab, cost_surface(E)

def stage_seg():
    rows, maps = [], {}
    variants = [("Full (SLIC + GMM + MRF-lite)", "full", {}), ("Without MRF-lite filter", "no_mrf", {}),
                ("Without superpixels (per-pixel GMM)", "pixel_gmm", {}), ("K-means + MRF-lite", "kmeans_mrf", {}),
                ("K-means baseline", "kmeans", {}), ("Full, k = 5 with two-highest-NDVI rule (earlier version)", "full", dict(k=5, rule="rank5"))]
    for s in SCENES:
        sc = make_scene(s); idx = indices(sc["img"])
        for nm, m, kw in variants:
            lab = segment(sc["img"], idx, m, **kw); a = accuracy(lab, sc["label"])
            rows.append(dict(scene=s, variant=nm, chi=coherence(lab), OA=a["OA"], kappa=a["kappa"], classes=a["classes_present"], recall=a["recall"], cm=a["cm"]))
            if s == REP: maps[nm] = lab
    sweep = []
    for div in [480, 240, 120, 60, 30]:
        for s in SCENES:
            sc = make_scene(s); idx = indices(sc["img"]); lab = segment(sc["img"], idx, "full", n_segments=int(220 * 220 / div))
            a = accuracy(lab, sc["label"]); sweep.append(dict(scene=s, div=div, chi=coherence(lab), OA=a["OA"], kappa=a["kappa"]))
    save(dict(rows=rows, maps=maps, sweep=sweep), "seg.pkl"); pd.DataFrame(rows).drop(columns=["cm"]).to_csv(os.path.join(OUT, "segmentation.csv"), index=False)

def stage_route(R):
    log, fronts = [], {}
    for si, s in enumerate(SCENES):
        sc, idx, E, lab, C = prep(s); st, gl = sc["start"], sc["goal"]
        def rec(solver, run, P, t, kind="route", extra=None):
            d = route_metrics(P, E, idx["NDVI"], lab); d.update(scene=s, solver=solver, run=run, time=t, kind=kind)
            if extra: d.update(extra)
            log.append(d)
        for nm, f in [("A*", astar), ("Dijkstra", dijkstra)]:
            f(C, st, gl); t = time.perf_counter(); P, info = f(C, st, gl); rec(nm, 0, P, time.perf_counter() - t, extra=dict(J=info["J"], expanded=info["expanded"]))
            fronts[(s, nm, 0)] = dict(route=P)
        for r in range(R):
            seed = 1000 * (si + 1) + r
            t = time.perf_counter(); P, info = aco(C, st, gl, lab == 4, seed=seed); rec("ACO", r, P, time.perf_counter() - t, extra=dict(J=info["J"]))
            fronts[(s, "ACO", r)] = dict(route=P)
            for nm, mem in [("NSGA-II", False), ("Memetic", True)]:
                t = time.perf_counter(); out = nsga2(C, E, st, gl, memetic=mem, seed=seed, record=True); dt = time.perf_counter() - t
                rec(nm, r, out["knee"], dt, "knee"); rec(nm, r, out["min_eco"], dt, "min_eco"); rec(nm, r, out["p_acc"], dt, "p_acc")
                fronts[(s, nm, r)] = dict(F=out["F"], hist=out["hist"], knee=out["knee"], min_eco=out["min_eco"], p_acc=out["p_acc"], front=out["front"] if (s == REP and r == 0) else None)
            print(s, r, flush=True)
        save(dict(log=log, fronts=fronts), "route.pkl"); pd.DataFrame(log).to_csv(os.path.join(OUT, "route_runs.csv"), index=False)

def stage_sens():
    sc, idx, E, lab, C = prep(REP); st, gl = sc["start"], sc["goal"]; out = {}
    base, _ = astar(C, st, gl); bset = set(base)
    jac = lambda P: len(set(P) & bset) / len(set(P) | bset)
    out["alpha"] = []
    for a in [0.1, 0.3, 0.6, 0.9, 1.2, 1.6, 2.0, 2.5, 3.0]:
        P, _ = astar(cost_surface(E, alpha=a), st, gl); d = route_metrics(P, E, idx["NDVI"], lab); d["alpha"] = a; out["alpha"].append(d)
    out["w1"] = []
    for w1 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        rest = np.array(DEFAULT_W[1:]); rest = rest / rest.sum() * (1 - w1)
        E2 = ess(idx, (w1, *rest)); P, _ = astar(cost_surface(E2), st, gl)
        out["w1"].append(dict(w1=w1, f_eco=f_eco(P, E), a_eco=a_eco(P, E), jac=jac(P)))
    rng = np.random.default_rng(7); dr = []
    for _ in range(300):
        w = rng.dirichlet(np.ones(4)); E2 = ess(idx, w); P, _ = astar(cost_surface(E2), st, gl)
        dr.append(dict(w=w.tolist(), jac=jac(P), f_eco=f_eco(P, E), a_eco=a_eco(P, E)))
    out["dirichlet"] = dr
    names = ["NDVI", "VARI", "ExG", "GLI"]; Z = np.stack([idx[k].ravel() for k in names], 1)
    out["corr"] = np.corrcoef(Z, rowvar=False).tolist()
    Zn = np.stack([minmax(idx[k]).ravel() for k in names], 1); vif = []
    for j in range(4):
        y = Zn[:, j]; X = np.c_[np.ones(len(y)), np.delete(Zn, j, 1)]; beta = np.linalg.lstsq(X, y, rcond=None)[0]
        r2 = 1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum(); vif.append(1 / max(1 - r2, 1e-9))
    out["vif"] = vif
    sl = minmax(slope(sc["dem"]), robust=False); out["layers"] = []
    for nm, kw in [("ESS only", {}), ("ESS + slope", dict(beta=0.6)), ("ESS + slope + protected", dict(beta=0.6, gamma=2.0)),
                   ("ESS + slope + protected + flood", dict(beta=0.6, gamma=2.0, delta=0.6))]:
        P, _ = astar(cost_surface(E, slope_n=sl, prot=sc["prot"], flood=sc["flood"], **kw), st, gl)
        r, c = np.asarray(P).T; d = route_metrics(P, E, idx["NDVI"], lab)
        d.update(name=nm, slope=float(sl[r, c].mean()), prot=int(sc["prot"][r, c].sum()), flood=int(sc["flood"][r, c].sum())); out["layers"].append(d)
    out["popgen"] = []
    for pop, gens in [(10, 45), (20, 45), (40, 45), (60, 45), (80, 45), (40, 10), (40, 20), (40, 30), (40, 60), (40, 80)]:
        for r in range(3):
            o = nsga2(C, E, st, gl, pop=pop, gens=gens, memetic=True, seed=500 + r)
            out["popgen"].append(dict(pop=pop, gens=gens, run=r, F=o["F"], min_eco=f_eco(o["min_eco"], E), knee_a=a_eco(o["knee"], E)))
        print("popgen", pop, gens, flush=True)
    out["base_route"] = base; out["scene"] = sc; out["E"] = E; out["idx"] = idx; out["lab"] = lab
    save(out, "sens.pkl")

def stage_eff():
    out = dict(mem=[], scal=[])
    sc, idx, E, lab, C = prep(REP); st, gl = sc["start"], sc["goal"]
    runs = [("A*", lambda s: astar(C, st, gl)), ("Dijkstra", lambda s: dijkstra(C, st, gl)), ("ACO", lambda s: aco(C, st, gl, lab == 4, seed=s)),
            ("NSGA-II", lambda s: nsga2(C, E, st, gl, seed=s)), ("Memetic", lambda s: nsga2(C, E, st, gl, memetic=True, seed=s))]
    for nm, f in runs:
        for r in range(3):
            tracemalloc.start(); f(900 + r); cur, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
            out["mem"].append(dict(solver=nm, peak_mb=peak / 2 ** 20))
    for size in [80, 120, 160, 220, 300]:
        sc, idx, E, lab, C = prep(REP, size=size); st, gl = sc["start"], sc["goal"]
        for nm, f in [("A*", lambda: astar(C, st, gl)), ("Dijkstra", lambda: dijkstra(C, st, gl)), ("ACO", lambda: aco(C, st, gl, lab == 4, seed=1)),
                      ("NSGA-II", lambda: nsga2(C, E, st, gl, seed=1)), ("Memetic", lambda: nsga2(C, E, st, gl, memetic=True, seed=1))]:
            t = time.perf_counter(); f(); out["scal"].append(dict(size=size, cells=size * size, solver=nm, time=time.perf_counter() - t))
        print("scal", size, flush=True)
    save(out, "eff.pkl")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", type=int, default=40); ap.add_argument("--stage", default="all")
    a = ap.parse_args(); t0 = time.time()
    if a.stage in ("all", "seg"): stage_seg(); print("seg done", time.time() - t0, flush=True)
    if a.stage in ("all", "sens"): stage_sens(); print("sens done", time.time() - t0, flush=True)
    if a.stage in ("all", "eff"): stage_eff(); print("eff done", time.time() - t0, flush=True)
    if a.stage in ("all", "route"): stage_route(a.runs); print("route done", time.time() - t0, flush=True)
