import numpy as np
import pytest

from brainlab import paths
from brainlab.store import db, graph
from brainlab.store.loaders import worm_cook2019 as worm


def _conn(tmp_db, worm_raw):
    conn = db.connect(tmp_db)
    worm.load(conn, worm_raw)
    return conn


def test_build_shapes_and_signs(tmp_db, worm_raw):
    conn = _conn(tmp_db, worm_raw)
    try:
        g = graph.build(conn, worm.DATASET_ID)
    finally:
        conn.close()
    n = g.n
    assert g.W_chem.shape == (n, n) and g.W_gap.shape == (n, n)
    assert n == len(g.names) == len(set(g.names))
    assert (g.W_gap != g.W_gap.T).nnz == 0                       # gap junctions are symmetric
    i = g.idx(["DD1"])[0]
    col = g.W_chem[:, i].toarray().ravel()
    assert col.min() < 0 and col.max() <= 0                       # everything from DD1 is inhibitory
    j = g.idx(["AVAL"])[0]
    assert g.W_chem[:, j].toarray().max() > 0


def test_gap_not_doubled(tmp_db, worm_raw):
    """Gap junctions are one entry per csv row, then symmetrized by taking the maximum
    (not the sum): W_gap[AVAL].sum() should match the sum of count over the raw rows where
    AVAL is pre and kind is electrical (one side), not the doubled value (it used to be 464)."""
    conn = _conn(tmp_db, worm_raw)
    try:
        g = graph.build(conn, worm.DATASET_ID)
        edges = db.edges(conn, worm.DATASET_ID)
    finally:
        conn.close()
    fwd = {e["post"]: e["count"] for e in edges if e["kind"] == "electrical" and e["pre"] == "AVAL"}
    bwd = {e["pre"]: e["count"] for e in edges if e["kind"] == "electrical" and e["post"] == "AVAL"}
    raw_total = sum(fwd.values())                          # one side — 232
    # almost all AVAL contacts are listed both ways with the same weight, but for the pair
    # AVAL-VA5/VA9 the weights are recorded differently in each direction (4 and 5) — symmetrizing by
    # the maximum takes the larger weight for each pair, so the exact value is a bit above raw_total,
    # but nowhere near double (as it was with the defect — 464).
    expected = sum(max(fwd.get(k, 0.0), bwd.get(k, 0.0)) for k in set(fwd) | set(bwd))
    i = g.idx(["AVAL"])[0]
    total = float(g.W_gap[i].sum())
    assert total == expected
    assert abs(total - raw_total) <= 2                      # not doubled, just a few asymmetric weights


def test_cache_roundtrip(tmp_db, worm_raw, tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "CACHE", tmp_path)
    conn = _conn(tmp_db, worm_raw)
    try:
        g = graph.build(conn, worm.DATASET_ID)
    finally:
        conn.close()
    graph.save_cache(g)
    g2 = graph.load_cache(worm.DATASET_ID)
    assert g2.names == g.names and g2.cell_class == g.cell_class
    assert (g2.W_chem != g.W_chem).nnz == 0 and (g2.W_gap != g.W_gap).nnz == 0
    assert np.array_equal(g.idx(["AVAL", "PLML"]), g2.idx(["AVAL", "PLML"]))
    assert g2.params == g.params and "sign_rule" in g2.params


def test_build_fills_params_from_dataset(tmp_db, worm_raw):
    """Graph.params — the dataset's params (datasets.params, task 1 of the final wave): for the worm
    this holds sign_rule; params survives build() and save_cache/load_cache."""
    conn = _conn(tmp_db, worm_raw)
    try:
        g = graph.build(conn, worm.DATASET_ID)
    finally:
        conn.close()
    assert "sign_rule" in g.params and g.params["sign_rule"]


def test_zero_sign_edges_not_explicit_in_nnz(tmp_db):
    """An edge with sign 0 (unknown transmitter) must not be an explicit zero in W_chem.nnz —
    eliminate_zeros() after building (task 5 of the final wave)."""
    conn = db.connect(tmp_db)
    db.register_dataset(conn, "toy0", "s", "1", "L", {}, {})
    db.add_neurons(conn, "toy0", [
        {"name": "A", "cell_type": "", "cell_class": "neuron", "transmitter": "", "side": "", "region": "", "extra": {}},
        {"name": "B", "cell_type": "", "cell_class": "neuron", "transmitter": "", "side": "", "region": "", "extra": {}}])
    db.add_edges(conn, "toy0", [
        {"pre": "A", "post": "B", "kind": "chemical", "count": 1, "sign": 0},
        {"pre": "B", "post": "A", "kind": "chemical", "count": 1, "sign": 1}])
    conn.commit()
    g = graph.build(conn, "toy0")
    conn.close()
    assert g.W_chem.nnz == 1     # the sign=0 edge does not count


def test_build_unknown_dataset_raises(tmp_db, worm_raw):
    conn = _conn(tmp_db, worm_raw)
    try:
        with pytest.raises(KeyError):
            graph.build(conn, "no_such_dataset")
    finally:
        conn.close()
