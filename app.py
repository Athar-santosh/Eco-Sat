"""Interactive screening application: streamlit run app.py"""
import numpy as np, streamlit as st, matplotlib.pyplot as plt
from vegsens.scenes import make_scene, SCENES
from vegsens.indices import indices, ess, cost_surface
from vegsens.segmentation import segment, coherence
from vegsens.solvers import astar, dijkstra, aco, nsga2
from vegsens.metrics import route_metrics
from vegsens import CLASSES

st.set_page_config(page_title="Vegetation-sensitivity corridor screening", layout="wide")
st.title("Label-free vegetation-sensitivity mapping and corridor screening")
with st.sidebar:
    src = st.radio("Input", ["Synthetic scene", "Upload 4-band .npy (H×W×4, R,G,B,NIR reflectance)"])
    name = st.selectbox("Scene", list(SCENES), index=2)
    alpha = st.slider("Eco-cost weight α", 0.1, 3.0, 1.2, 0.1)
    solver = st.selectbox("Solver", ["A*", "Dijkstra", "ACO", "NSGA-II", "Memetic (A*-seeded NSGA-II)"])
    pop = st.slider("NSGA-II population", 10, 80, 40, 10); gens = st.slider("Generations", 10, 80, 45, 5)
    seed = st.number_input("Random seed", 0, 10_000, 0)
sc = make_scene(name)
if src.startswith("Upload"):
    f = st.file_uploader("4-band array", type=["npy"])
    if f is None: st.stop()
    img = np.load(f).astype(float); sc.update(img=img, start=(int(.18 * img.shape[0]), 0), goal=(int(.82 * img.shape[0]), img.shape[1] - 1))
img = sc["img"]; idx = indices(img); E = ess(idx); lab = segment(img, idx, "full"); C = cost_surface(E, alpha=alpha)
s, g = sc["start"], sc["goal"]
if solver == "A*": P, _ = astar(C, s, g); front = None
elif solver == "Dijkstra": P, _ = dijkstra(C, s, g); front = None
elif solver == "ACO": P, _ = aco(C, s, g, lab == 4, seed=int(seed)); front = None
else:
    out = nsga2(C, E, s, g, pop=pop, gens=gens, memetic=solver.startswith("Memetic"), seed=int(seed))
    choice = st.sidebar.radio("Alignment from front", ["knee", "min_eco", "p_acc"]); P = out[choice]; front = out["F"]
c1, c2, c3 = st.columns(3)
cmap = plt.matplotlib.colors.ListedColormap(["#2b6cb0", "#a0a0a0", "#c8a46e", "#9acd32", "#1e5b1e"])
def show(col, arr, title, cm, path=None):
    fig, ax = plt.subplots(); ax.imshow(arr, cmap=cm); ax.set_title(title); ax.axis("off")
    if path is not None: p = np.asarray(path); ax.plot(p[:, 1], p[:, 0], "r-", lw=1.5)
    col.pyplot(fig)
show(c1, lab, f"Classified map (χ = {coherence(lab):.3f})", cmap)
show(c2, E, "Vegetation ESS", "magma")
show(c3, idx["NDVI"], "Route on NDVI", "Greens", P)
st.subheader("Route metrics"); st.json({k: round(v, 4) if isinstance(v, float) else v for k, v in route_metrics(P, E, idx["NDVI"], lab).items()})
st.subheader("Per-class area (%)"); st.bar_chart({c: float((lab == i).mean() * 100) for i, c in enumerate(CLASSES)})
if front is not None:
    fig, ax = plt.subplots(); o = np.argsort(front[:, 1]); ax.plot(front[o, 1], front[o, 0], "o-"); ax.set_xlabel("length (px)"); ax.set_ylabel("mean ESS"); st.pyplot(fig)
