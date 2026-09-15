import json

import pytest
import yaml

from brainlab import paths
from brainlab.lab import atlas


def test_atlas_table_from_fake_runner(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RESULTS", tmp_path / "results"); monkeypatch.setattr(paths, "LAB", tmp_path / "lab")
    (tmp_path / "lab" / "experiments").mkdir(parents=True)
    for ds in ("d1", "d2"):
        (tmp_path / "lab" / "experiments" / ("taste_%s.yaml" % ds)).write_text(yaml.safe_dump({
            "name": "taste_%s" % ds, "dataset": ds, "model": "lif", "duration_ms": 10, "stimulus": {"pulses": []},
            "measure": {"sugar": {"group": "mn9", "window": [0, 10]}}}), encoding="utf-8")

    class R:
        def __init__(self, ds): self.id = "run-" + ds; self.valence = 1; self.flags = {"silent": False}; self.extra = {"empty_groups": []}
        def hz(self, names, t0, t1): return 42.0 if names else 0.0
    monkeypatch.setattr(atlas.runner, "run_experiment", lambda p, save=True: R(yaml.safe_load(open(p, encoding="utf-8"))["dataset"]))
    monkeypatch.setattr(atlas.populations, "names", lambda ds, g: ["x"] if ds == "d1" else [])

    class _FakeConn:
        def close(self): pass

    def fake_dataset_info(conn, ds):
        return {"version": "v-" + ds, "params": json.dumps({"min_count": 5, "sign_rule": "rule-" + ds})}
    monkeypatch.setattr(atlas.db, "connect", lambda: _FakeConn())
    monkeypatch.setattr(atlas.db, "dataset_info", fake_dataset_info)
    out = atlas.run("taste", ["d1", "d2"], save=False)
    assert out["d1"]["sugar"] == 42.0 and out["d2"]["sugar"] == 0.0
    assert out["d1"]["version"] == "v-d1" and out["d1"]["min_count"] == 5 and out["d1"]["sign_rule"] == "rule-d1"
    md = (tmp_path / "results" / "atlas" / "taste.md").read_text(encoding="utf-8")
    assert "d1" in md and "42.0" in md and "v-d1" in md and "rule-d1" in md
    assert "min_count=5" in md    # the threshold/rule caveat under the table
    assert json.loads((tmp_path / "results" / "atlas" / "taste.json").read_text(encoding="utf-8"))["d2"]["sugar"] == 0.0
    atlas.run("taste", ["d1"], save=False)
    assert (tmp_path / "results" / "atlas" / "taste-1.md").exists()     # the old table was not overwritten


def test_atlas_rejects_dataset_filename_mismatch(tmp_path, monkeypatch):
    """yaml dataset: must match the file token <name>_<token>.yaml — otherwise measure would silently
    read groups from the wrong dataset (dataset from the spec vs ds from the filename)."""
    monkeypatch.setattr(paths, "RESULTS", tmp_path / "results"); monkeypatch.setattr(paths, "LAB", tmp_path / "lab")
    (tmp_path / "lab" / "experiments").mkdir(parents=True)
    (tmp_path / "lab" / "experiments" / "taste_d1.yaml").write_text(yaml.safe_dump({
        "name": "taste_d1", "dataset": "d2", "model": "lif", "duration_ms": 10, "stimulus": {"pulses": []},
        "measure": {"sugar": {"group": "mn9", "window": [0, 10]}}}), encoding="utf-8")

    class R:
        def __init__(self): self.id = "run-x"; self.valence = 1; self.flags = {"silent": False}; self.extra = {"empty_groups": []}
        def hz(self, names, t0, t1): return 42.0
    monkeypatch.setattr(atlas.runner, "run_experiment", lambda p, save=True: R())
    monkeypatch.setattr(atlas.populations, "names", lambda ds, g: ["x"])
    with pytest.raises(ValueError):
        atlas.run("taste", ["d1"], save=False)


def test_atlas_accepts_token_starting_with_dataset(tmp_path, monkeypatch):
    """Token banc_888_norm -> file <name>_banc_888_norm.yaml with dataset: banc_888 — is allowed:
    the check is relaxed to "the token starts with dataset" (a normalized variant of the same dataset)."""
    monkeypatch.setattr(paths, "RESULTS", tmp_path / "results"); monkeypatch.setattr(paths, "LAB", tmp_path / "lab")
    (tmp_path / "lab" / "experiments").mkdir(parents=True)
    (tmp_path / "lab" / "experiments" / "taste_d1_norm.yaml").write_text(yaml.safe_dump({
        "name": "taste_d1_norm", "dataset": "d1", "model": "lif", "duration_ms": 10, "stimulus": {"pulses": []},
        "graph": {"normalize": "d0"}, "measure": {"sugar": {"group": "mn9", "window": [0, 10]}}}), encoding="utf-8")

    class R:
        id = "run-x"; valence = 1; flags = {}; extra = {}
        def hz(self, names, t0, t1): return 1.0
    monkeypatch.setattr(atlas.runner, "run_experiment", lambda p, save=True: R())
    monkeypatch.setattr(atlas.populations, "names", lambda ds, g: ["x"])

    class _FakeConn:
        def close(self): pass
    monkeypatch.setattr(atlas.db, "connect", lambda: _FakeConn())
    monkeypatch.setattr(atlas.db, "dataset_info", lambda conn, ds: {"version": "v", "params": "{}"})
    out = atlas.run("taste", ["d1_norm"], save=False)
    assert out["d1_norm"]["sugar"] == 1.0
