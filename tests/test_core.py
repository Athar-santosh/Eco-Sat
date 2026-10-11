import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from vegsens.scenes import make_scene
from vegsens.indices import indices, ess, cost_surface
from vegsens.solvers import astar, dijkstra, crossover, remove_loops
from vegsens.metrics import hypervolume_2d

def test_astar_equals_dijkstra_cost():
    sc = make_scene("S3"); C = cost_surface(ess(indices(sc["img"])))
    _, a = astar(C, sc["start"], sc["goal"]); _, d = dijkstra(C, sc["start"], sc["goal"])
    assert abs(a["J"] - d["J"]) < 1e-6

def test_crossover_keeps_connectivity():
    rng = np.random.default_rng(0)
    p1 = [(0, i) for i in range(10)]; p2 = [(0, 0), (1, 1), (1, 2), (0, 3), (0, 4), (1, 5), (2, 6), (1, 7), (0, 8), (0, 9)]
    for _ in range(20):
        for c in crossover(p1, p2, rng):
            assert all(max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1 for a, b in zip(c[:-1], c[1:]))

def test_remove_loops():
    assert remove_loops([(0, 0), (0, 1), (1, 1), (0, 1), (0, 2)]) == [(0, 0), (0, 1), (0, 2)]

def test_hypervolume():
    assert abs(hypervolume_2d([[0, 0]], (1, 1)) - 1) < 1e-12
