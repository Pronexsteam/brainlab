import numpy as np

from brainlab import paths
from brainlab.sim import flags, result, stimulus
from brainlab.store import db, graph
from brainlab.store.loaders import worm_cook2019 as worm


def test_stimulus_drive(tmp_db, worm_raw):
    conn = db.connect(tmp_db); worm.load(conn, worm_raw)
    g = graph.build(conn, worm.DATASET_ID)
    st = stimulus.Stimulus([stimulus.Pulse(["ALML", "ALMR"], 1.0, 100, 200), stimulus.Pulse(["AVM"], 0.5, 150, 250)])
    d = st.drive(g, 50)
    assert d.sum() == 0
    d = st.drive(g, 175)
    assert d[g.idx(["ALML"])[0]] == 1.0 and d[g.idx(["AVM"])[0]] == 0.5 and d.sum() == 2.5
    assert stimulus.Stimulus.from_dict(st.to_dict()).to_dict() == st.to_dict()


def test_pulse_kind_roundtrip_and_compile():
    from brainlab.sim import stimulus
    st = stimulus.Stimulus([stimulus.Pulse(["a"], 150.0, 100, 300, kind="poisson"), stimulus.Pulse(["b"], 1.0, 0, 50)])
    d = st.to_dict()
    assert d["pulses"][0]["kind"] == "poisson" and d["pulses"][1]["kind"] == "current"
    back = stimulus.Stimulus.from_dict({"pulses": [{"names": ["a"], "value": 1.0, "t0_ms": 0, "t1_ms": 10}]})
    assert back.pulses[0].kind == "current"

    class G:  # минимальный граф для compile
        names = ["a", "b"]; n = 2
        def idx(self, names): return __import__("numpy").array([self.names.index(x) for x in names])
    comp = st.compile(G(), dt_ms=0.5)
    assert [(c.k0, c.k1, c.kind) for c in comp] == [(200, 600, "poisson"), (0, 100, "current")]


def test_result_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RESULTS", tmp_path)
    rates = np.random.default_rng(0).random((4, 3)).astype(np.float32)
    r = result.RunResult("toy", "graded", {"tau_ms": 50}, 7, {"pulses": [], "noise": 0.0}, 1.0, 50.0,
                         ["A", "B", "C"], rates, {"silent": False}, result.code_hash())
    p = r.save()
    assert (p / "run.json").exists() and (p / "rates.npz").exists() and (p / "summary.md").exists()
    r2 = result.load(p)
    assert r2.params == {"tau_ms": 50} and r2.seed == 7 and r2.names == ["A", "B", "C"]
    assert np.array_equal(r2.rates, rates) and r2.code_hash == r.code_hash and len(r.code_hash) == 64
    ids = [x["id"] for x in result.list_runs()]
    assert p.name in ids


def test_flags():
    quiet = np.zeros((10, 5), np.float32)
    assert flags.check(quiet, 50.0)["silent"] is True
    burst = np.full((10, 5), 0.99, np.float32)
    assert flags.check(burst, 50.0)["seizure"] is True
    ok = np.random.default_rng(1).random((10, 5)).astype(np.float32) * 0.3
    f = flags.check(ok, 50.0)
    assert not f["silent"] and not f["seizure"]
