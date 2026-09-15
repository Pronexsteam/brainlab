import json

import yaml

from brainlab import paths
from brainlab.lab import runner


def test_group_stimulus_expands_to_names(worm_raw, tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RESULTS", tmp_path)
    spec = {
        "name": "group_expand_smoke", "dataset": "worm_cook2019", "model": "graded",
        "duration_ms": 200, "stimulus": {"pulses": [{"names": "group:posterior_touch", "value": 1.0, "t0_ms": 0, "t1_ms": 50}]},
    }
    spec_path = tmp_path / "group_expand_smoke.yaml"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    r = runner.run_experiment(spec_path, save=True)
    assert r.stimulus["pulses"][0]["names"] == ["PLML", "PLMR"]
    assert r.extra["experiment_groups"] == {0: "group:posterior_touch"}
    saved = json.loads((tmp_path / r.id / "run.json").read_text(encoding="utf-8"))
    assert saved["stimulus"]["pulses"][0]["names"] == ["PLML", "PLMR"]
    assert saved["extra"]["experiment_groups"] == {"0": "group:posterior_touch"}


def test_evaluate_no_response_on_first_is_not_pain(monkeypatch):
    """first ≈ 0 (нет ответа на первый стимул) — не боль (ratio=inf раньше ложно проходил
    pain_if_above), а честный no_response=True и valence=0."""
    class R:
        dataset = "toy"
        flags = {}
        def group_rate(self, names, t0, t1):
            return 0.0 if t0 == 0 else 5.0
    monkeypatch.setattr(runner.populations, "names", lambda ds, g: ["x"])
    spec = {"valence": {"group": "g", "first": [0, 10], "last": [100, 110], "pain_if_above": 2.0, "reward_if_below": -1}}
    out = runner.evaluate(R(), spec)
    assert out["no_response"] is True
    assert out["valence"] == 0
    assert out["ratio"] is None            # None, не inf: json/JS не знают Infinity


def test_habituation_experiment(worm_raw, tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RESULTS", tmp_path)
    spec_path = paths.LAB / "experiments" / "worm_tap_habituation.yaml"
    spec = runner.load_experiment(spec_path)
    assert spec["model"] == "graded" and len(spec["stimulus"]["pulses"]) == 11
    r = runner.run_experiment(spec_path, save=True)
    ev = runner.evaluate(r, spec)
    print(ev)
    assert ev["ratio"] < 0.8, ev                  # 10-е постукивание слабее первого
    assert ev["novel"] > ev["last"]               # новый стимул возвращает ответ
    assert r.valence == 1 and r.id and (tmp_path / r.id / "run.json").exists()


def test_graph_normalize_key_passed_to_graph_get(tmp_path, monkeypatch):
    """Ключ graph: {normalize: <ref>} в YAML опыта уходит в graph_mod.get(ds, normalize=...)
    и записывается в extra["graph"]."""
    calls = {}

    class G:
        n = 2; names = ["a", "b"]; dataset = "toy"; version = "v"; params = {"normalize": {"ref": "r"}}
        index = {"a": 0, "b": 1}
        def idx(self, names): return [self.index[n] for n in names]

    class M:
        def __init__(self, g, seed=0, **kw): self.g = g
        def run(self, st, dur, win):
            class R:
                dataset = "toy"; flags = {}; extra = {}; valence = 0; stimulus = {}
                def save(self): pass
            return R()
    monkeypatch.setattr(runner.graph_mod, "get", lambda ds, conn=None, normalize=None: calls.setdefault("args", (ds, normalize)) or G())
    monkeypatch.setitem(runner.MODELS, "fake", M)
    monkeypatch.setattr(runner.state_mod, "attach", lambda r, ds, names=None: None)
    spec = {"name": "norm_smoke", "dataset": "toy", "model": "fake", "duration_ms": 1,
            "stimulus": {"pulses": []}, "graph": {"normalize": "r"}}
    p = tmp_path / "norm_smoke.yaml"
    p.write_text(yaml.safe_dump(spec), encoding="utf-8")
    r = runner.run_experiment(p, save=False)
    assert calls["args"] == ("toy", "r")
    assert r.extra["graph"] == {"normalize": "r"}
