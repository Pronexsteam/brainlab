import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import selftest  # noqa: E402


def test_selftest_calls_expected_gates_and_runner():
    text = (ROOT / "selftest.py").read_text(encoding="utf-8")
    assert "gate1_worm_touch.run" in text
    assert "gate2_fly_sugar.run" in text
    assert "runner.run_experiment" in text


def _write_run(folder, dataset_version="", code_hash="deadbeef", extra=None, write_rates=True):
    folder.mkdir(parents=True, exist_ok=True)
    meta = {
        "dataset": "worm_cook2019", "model": "graded", "params": {}, "seed": 0,
        "stimulus": {"pulses": []}, "dt_ms": 1.0, "window_ms": 50.0, "names": [],
        "flags": {}, "code_hash": code_hash, "valence": 0, "parent": "", "archived": False,
        "id": folder.name, "extra": extra if extra is not None else {}, "dataset_version": dataset_version,
    }
    (folder / "run.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    if write_rates:
        (folder / "rates.npz").write_bytes(b"")


def test_honesty_red_when_dataset_version_missing_after_cutoff(tmp_path):
    _write_run(tmp_path / "20260914-225959-00", dataset_version="")
    ok, report = selftest.honesty(tmp_path)
    assert ok is False
    assert report["прогонов без dataset_version (новее рубежа %s)" % selftest.DATASET_VERSION_CUTOFF] == [
        "20260914-225959-00"
    ]


def test_honesty_green_when_dataset_version_missing_before_cutoff(tmp_path):
    _write_run(tmp_path / "20260914-215000-00", dataset_version="")
    ok, report = selftest.honesty(tmp_path)
    assert ok is True
    assert report["прогонов без dataset_version (новее рубежа %s)" % selftest.DATASET_VERSION_CUTOFF] == []


def test_honesty_green_when_dataset_version_present(tmp_path):
    _write_run(tmp_path / "20260915-010000-00", dataset_version="cook2019-c302")
    ok, report = selftest.honesty(tmp_path)
    assert ok is True


def test_honesty_red_when_rates_without_run_json(tmp_path):
    folder = tmp_path / "20260915-020000-00"
    folder.mkdir(parents=True)
    (folder / "rates.npz").write_bytes(b"")
    ok, report = selftest.honesty(tmp_path)
    assert ok is False
    assert report["прогонов без run.json (есть rates.npz)"] == ["20260915-020000-00"]


def test_honesty_red_when_code_hash_empty(tmp_path):
    _write_run(tmp_path / "20260915-020000-00", dataset_version="cook2019-c302", code_hash="")
    ok, report = selftest.honesty(tmp_path)
    assert ok is False
    assert report["прогонов без code_hash"] == ["20260915-020000-00"]


def test_honesty_red_when_extreme_flag_not_bool(tmp_path):
    _write_run(tmp_path / "20260915-020000-00", dataset_version="cook2019-c302",
                extra={"extreme": "true"})
    ok, report = selftest.honesty(tmp_path)
    assert ok is False
    assert report["прогонов с некорректным extra.extreme"] == ["20260915-020000-00"]


def test_honesty_red_when_atlas_row_missing_run_id(tmp_path):
    _write_run(tmp_path / "20260915-020000-00", dataset_version="cook2019-c302")
    atlas_dir = tmp_path / "atlas"
    atlas_dir.mkdir()
    (atlas_dir / "x.json").write_text(
        json.dumps({"fafb_783": {"valence": 1, "sugar": 10.0}}, ensure_ascii=False), encoding="utf-8")
    ok, report = selftest.honesty(tmp_path)
    assert ok is False
    assert report["строк атласа без run_id"] == ["x.json:fafb_783"]
