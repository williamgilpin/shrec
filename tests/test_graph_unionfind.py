"""Parity oracle for the hand-rolled union-find (docs/tests-math.md §5b.1 MM11).

``graph/unionfind.py`` predates ``scipy.cluster.hierarchy.DisjointSet`` and is
kept for the ``ClassicalRecurrenceClustering`` merge step. MM11 pins that it
computes the *same connected-components partition* as scipy's reference
implementation on random merge sequences, so the legacy code could be swapped
for scipy behind an adapter without behaviour change.
"""
import numpy as np
import pytest
from scipy.cluster.hierarchy import DisjointSet as ScipyDisjointSet

from shrec.graph.unionfind import DisjointSet, solve_union_find


def _partition_ours(n, edges):
    ds = DisjointSet()
    for node in range(n):
        ds.find(node)  # register singletons so the partition covers all nodes
    for a, b in edges:
        ds.union(a, b)
    return {frozenset(members) for members in ds.groups().values()}


def _partition_scipy(n, edges):
    ds = ScipyDisjointSet(range(n))
    for a, b in edges:
        ds.merge(a, b)
    return {frozenset(s) for s in ds.subsets()}


class TestUnionFindParity:

    @pytest.mark.parametrize("seed", range(8))
    def test_partition_matches_scipy(self, seed):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(8, 20))
        m = int(rng.integers(1, 2 * n))
        edges = rng.integers(0, n, size=(m, 2)).tolist()
        assert _partition_ours(n, edges) == _partition_scipy(n, edges)

    def test_all_singletons_when_no_merges(self):
        assert _partition_ours(6, []) == {frozenset({i}) for i in range(6)}

    def test_transitive_chain_is_one_class(self):
        # 0-1, 1-2, 2-3 ⇒ a single equivalence class.
        assert _partition_ours(4, [(0, 1), (1, 2), (2, 3)]) == {frozenset({0, 1, 2, 3})}


class TestSolveUnionFind:
    """`solve_union_find` returns, per input group, the full equivalence class
    of that group's first element. Cross-check the closure against scipy."""

    @pytest.mark.parametrize("seed", range(8))
    def test_returned_class_is_scipy_closure(self, seed):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(8, 20))
        n_groups = int(rng.integers(2, 6))
        groups = [
            rng.integers(0, n, size=int(rng.integers(1, 4))).tolist()
            for _ in range(n_groups)
        ]

        ds = ScipyDisjointSet(range(n))
        for g in groups:
            for x in g[1:]:
                ds.merge(g[0], x)

        result = solve_union_find([list(g) for g in groups])
        for g, cls in zip(groups, result):
            expected = ds.subset(g[0])
            assert set(cls) == set(expected)
