import pytest

from brainlab.store import db
from brainlab.store.loaders import worm_cook2019 as worm


def test_load_worm(tmp_db, worm_raw):
    conn = db.connect(tmp_db)
    summary = worm.load(conn, worm_raw)
    assert 295 <= summary["neurons"] <= 302, summary
    assert summary["other"] > 50                      # muscles and end organs are in the diagram too
    assert summary["edges"] > 6000
    assert 20 <= summary["gaba"] <= 30
    names = {n["name"] for n in db.neurons(conn, worm.DATASET_ID)}
    for must in ("AVAL", "AVAR", "AVBL", "AVBR", "AVDL", "AVDR", "AVEL", "AVER",
                 "PVCL", "PVCR", "ALML", "ALMR", "AVM", "PLML", "PLMR", "I1L", "NSML"):
        assert must in names, must
    assert all(n == n.strip() for n in names)         # leading/trailing whitespace is stripped
    kinds = {e["kind"] for e in db.edges(conn, worm.DATASET_ID)}
    assert kinds == {"chemical", "electrical"}
    gaba_edges = [e for e in db.edges(conn, worm.DATASET_ID) if e["pre"] == "DD1" and e["kind"] == "chemical"]
    assert gaba_edges and all(e["sign"] == -1 for e in gaba_edges)
    info = db.dataset_info(conn, worm.DATASET_ID)
    assert "herm_full_edgelist.csv" in info["files"] and "sign_rule" in info["params"]


def test_load_rolls_back_on_failing_add_edges(tmp_db, worm_raw, monkeypatch):
    """The loader is one transaction (register_dataset+add_neurons+add_edges, task 5 of the final
    wave): if add_edges fails, the dataset is absent from the database entirely (not half-loaded)."""
    conn = db.connect(tmp_db)

    def boom(*a, **kw):
        raise RuntimeError("failure while inserting edges")
    monkeypatch.setattr(db, "add_edges", boom)
    with pytest.raises(RuntimeError):
        worm.load(conn, worm_raw)
    assert db.dataset_info(conn, worm.DATASET_ID) is None
    assert db.neurons(conn, worm.DATASET_ID) == []
