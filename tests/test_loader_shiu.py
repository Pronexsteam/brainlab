import pandas as pd
import pytest

from brainlab import paths
from brainlab.store import db, graph
from brainlab.store.loaders import fly_shiu630 as shiu


def _toy(tmp_path):
    raw = tmp_path / "raw"; raw.mkdir()
    pd.DataFrame({"Unnamed: 0": [1001, 1002, 1003], "Completed": [True, True, True]}).to_csv(
        raw / "2023_03_23_completeness_630_final.csv", index=False)
    con = pd.DataFrame({"Presynaptic_ID": [1001, 1001, 1002], "Postsynaptic_ID": [1002, 1003, 1003],
                        "Presynaptic_Index": [0, 0, 1], "Postsynaptic_Index": [1, 2, 2],
                        "Connectivity": [5, 2, 7], "Excitatory": [1, 1, -1]})
    con["Excitatory x Connectivity"] = con["Connectivity"] * con["Excitatory"]
    con.to_parquet(raw / "2023_03_23_connectivity_630_final.parquet")
    return raw


def test_load_toy(tmp_db, tmp_path):
    conn = db.connect(tmp_db)
    s = shiu.load(conn, _toy(tmp_path))
    assert s == {"neurons": 3, "edges": 3, "positive": 2, "negative": 1}
    ns = {n["name"]: n for n in db.neurons(conn, shiu.DATASET_ID)}
    assert ns["1001"]["transmitter"] == "exc" and ns["1002"]["transmitter"] == "inh" and ns["1003"]["transmitter"] == ""
    g = graph.build(conn, shiu.DATASET_ID)
    assert g.W_chem[g.index["1003"], g.index["1002"]] == -7 and g.W_chem[g.index["1002"], g.index["1001"]] == 5
    assert g.version == "630-shiu-2023_03_23" and g.W_gap.nnz == 0
    conn.close()


@pytest.mark.slow
def test_load_real(tmp_db):
    raw = paths.DATA / "flywire_630_shiu" / "raw"
    if not (raw / "2023_03_23_connectivity_630_final.parquet").exists():
        pytest.skip("нет сырья эталона")
    conn = db.connect(tmp_db)
    s = shiu.load(conn, raw)
    assert s["neurons"] == 127400 and s["edges"] == 14687178
    conn.close()
