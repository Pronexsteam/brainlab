from brainlab import paths
from brainlab.lab.gates import gate1_worm_touch as gate
from brainlab.store import db, graph
from brainlab.store.loaders import worm_cook2019 as worm


def test_gate1_touch_asymmetry(tmp_db, worm_raw, tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RESULTS", tmp_path)
    conn = db.connect(tmp_db); worm.load(conn, worm_raw)
    g = graph.build(conn, worm.DATASET_ID)
    out = gate.run(g, seed=0, save=True)
    print(out)
    assert out["anterior_back"] > out["anterior_fwd"], out
    assert out["posterior_fwd"] > out["posterior_back"], out
    assert out["passed"] is True and len(out["run_ids"]) == 2
