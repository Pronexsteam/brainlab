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
    assert (g.W_gap != g.W_gap.T).nnz == 0                       # щелевые контакты симметричны
    i = g.idx(["DD1"])[0]
    col = g.W_chem[:, i].toarray().ravel()
    assert col.min() < 0 and col.max() <= 0                       # всё, что идёт от DD1, тормозное
    j = g.idx(["AVAL"])[0]
    assert g.W_chem[:, j].toarray().max() > 0


def test_gap_not_doubled(tmp_db, worm_raw):
    """Щелевые контакты — одна запись на строку csv, потом симметризация через максимум
    (не сумму): W_gap[AVAL].sum() должна совпадать с суммой count по сырым строкам, где
    AVAL — pre и kind электрический (одна сторона), а не удвоенным значением (было 464)."""
    conn = _conn(tmp_db, worm_raw)
    try:
        g = graph.build(conn, worm.DATASET_ID)
        edges = db.edges(conn, worm.DATASET_ID)
    finally:
        conn.close()
    fwd = {e["post"]: e["count"] for e in edges if e["kind"] == "electrical" and e["pre"] == "AVAL"}
    bwd = {e["pre"]: e["count"] for e in edges if e["kind"] == "electrical" and e["post"] == "AVAL"}
    raw_total = sum(fwd.values())                          # одна сторона — 232
    # почти все контакты AVAL перечислены в обе стороны с одинаковым весом, но у пары
    # AVAL-VA5/VA9 веса записаны по-разному в разные стороны (4 и 5) — симметризация через
    # максимум берёт больший вес по каждой паре, поэтому точное значение чуть выше raw_total,
    # но далеко не вдвое (как было при дефекте — 464).
    expected = sum(max(fwd.get(k, 0.0), bwd.get(k, 0.0)) for k in set(fwd) | set(bwd))
    i = g.idx(["AVAL"])[0]
    total = float(g.W_gap[i].sum())
    assert total == expected
    assert abs(total - raw_total) <= 2                      # не удвоение, лишь редкие асимметричные веса


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
    """Graph.params — параметры набора (datasets.params, задача 1 финальной волны): у червя
    там sign_rule; params переживает build() и save_cache/load_cache."""
    conn = _conn(tmp_db, worm_raw)
    try:
        g = graph.build(conn, worm.DATASET_ID)
    finally:
        conn.close()
    assert "sign_rule" in g.params and g.params["sign_rule"]


def test_zero_sign_edges_not_explicit_in_nnz(tmp_db):
    """Ребро со знаком 0 (неизвестный медиатор) не должно быть явным нулём в W_chem.nnz —
    eliminate_zeros() после сборки (задача 5 финальной волны)."""
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
    assert g.W_chem.nnz == 1     # ребро sign=0 не считается


def test_build_unknown_dataset_raises(tmp_db, worm_raw):
    conn = _conn(tmp_db, worm_raw)
    try:
        with pytest.raises(KeyError):
            graph.build(conn, "нет_такого")
    finally:
        conn.close()
