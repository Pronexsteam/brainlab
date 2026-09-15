import numpy as np

from brainlab.store import db, graph
from brainlab.store.loaders import worm_cook2019 as worm


def _build_slow(conn, ds):
    """Старый способ сборки — через список словарей; служит эталоном для новой сборки."""
    import scipy.sparse as sp
    ns = db.neurons(conn, ds)
    names = [n["name"] for n in ns]
    index = {n: i for i, n in enumerate(names)}
    rc, cc, vc, rg, cg, vg = [], [], [], [], [], []
    for e in db.edges(conn, ds):
        a, b = index[e["pre"]], index[e["post"]]
        if e["kind"] == "electrical":
            rg.append(b); cg.append(a); vg.append(e["count"])
        else:
            rc.append(b); cc.append(a); vc.append(e["sign"] * e["count"])
    n = len(names)
    Wc = sp.csr_matrix((vc, (rc, cc)), shape=(n, n), dtype=np.float32)
    Wg = sp.csr_matrix((vg, (rg, cg)), shape=(n, n), dtype=np.float32)
    Wg.sum_duplicates()
    return Wc, Wg.maximum(Wg.T).tocsr()


def test_array_build_matches_dict_build(tmp_db, worm_raw):
    conn = db.connect(tmp_db)
    worm.load(conn, worm_raw)
    g = graph.build(conn, worm.DATASET_ID)
    Wc, Wg = _build_slow(conn, worm.DATASET_ID)
    conn.close()
    assert (g.W_chem != Wc).nnz == 0
    assert (g.W_gap != Wg).nnz == 0
    assert g.W_chem.dtype == np.float32


def test_add_edges_chunks(tmp_db):
    conn = db.connect(tmp_db)
    db.register_dataset(conn, "toy", "", "1", "", {}, {})
    db.add_neurons(conn, "toy", [{"name": "a"}, {"name": "b"}])
    rows = [{"pre": "a", "post": "b", "kind": "chemical", "count": i, "sign": 1} for i in range(1, 1001)]
    db.add_edges(conn, "toy", rows, chunk=300)
    assert conn.execute("SELECT COUNT(*) FROM edges WHERE dataset='toy'").fetchone()[0] == 1000
    conn.close()
