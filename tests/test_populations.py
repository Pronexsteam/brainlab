import json
import os

import yaml

from brainlab import paths
from brainlab.lab import populations
from brainlab.store import db


def _toy_db(tmp_db):
    conn = db.connect(tmp_db)
    db.register_dataset(conn, "toy", "", "1", "", {}, {})
    db.add_neurons(conn, "toy", [
        {"name": "1", "cell_type": "MN9", "cell_class": "motor", "side": "right", "extra": {"cell_function_detailed": "sugar, Gr64f"}},
        {"name": "2", "cell_type": "MN9", "cell_class": "motor", "side": "left", "extra": {}},
        {"name": "3", "cell_type": "PAM01", "cell_class": "central_brain_intrinsic", "side": "left", "extra": {"fafb_cell_type": "PAM01"}},
        {"name": "4", "cell_type": "", "cell_class": "sensory", "side": "left", "extra": {"cell_function_detailed": "bitter, Gr33a"}}])
    conn.commit()   # register_dataset/add_neurons больше не коммитят сами (задача 5 финальной волны)
    return conn


def test_resolve_rules(tmp_db, tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DB_PATH", tmp_db); monkeypatch.setattr(paths, "LAB", tmp_path); monkeypatch.setattr(paths, "CACHE", tmp_path / "cache")
    conn = _toy_db(tmp_db); conn.close()
    (tmp_path / "populations").mkdir()
    (tmp_path / "populations" / "toy.yaml").write_text(yaml.safe_dump({"dataset": "toy", "groups": {
        "mn9": {"where": {"cell_type": "MN9"}}, "mn9_right": {"where": {"cell_type": "MN9", "side": "right"}},
        "sugar": {"where": {"cell_function_detailed": {"startswith": "sugar"}}},
        "dan": {"where": {"any": [{"cell_type": {"startswith": "PAM"}}, {"fafb_cell_type": {"startswith": "PAM"}}]}},
        "explicit": {"names": ["4", "9"]}, "empty": {"where": {"cell_type": "нет"}}},
        "state": ["mn9", "sugar"]}), encoding="utf-8")
    r = populations.resolve("toy")
    assert r["mn9"] == ["1", "2"] and r["mn9_right"] == ["1"] and r["sugar"] == ["1"] and r["dan"] == ["3"]
    assert r["explicit"] == ["4"] and r["empty"] == []           # неизвестное имя 9 отброшено, пустая группа допустима
    assert populations.state_groups("toy") == ["mn9", "sugar"]
    assert (tmp_path / "cache" / "toy_populations.json").exists()
    assert populations.names("toy", "mn9") == ["1", "2"]


def test_resolve_tolerates_incomplete_cache(tmp_db, tmp_path, monkeypatch):
    """Кэш с правильным yaml_mtime, но без "groups" (обрезанная/битая запись) —
    resolve() должен пересчитать заново, а не падать и не отдавать мусор."""
    monkeypatch.setattr(paths, "DB_PATH", tmp_db); monkeypatch.setattr(paths, "LAB", tmp_path); monkeypatch.setattr(paths, "CACHE", tmp_path / "cache")
    conn = _toy_db(tmp_db); conn.close()
    (tmp_path / "populations").mkdir()
    (tmp_path / "populations" / "toy.yaml").write_text(yaml.safe_dump({"dataset": "toy", "groups": {
        "mn9": {"where": {"cell_type": "MN9"}}}, "state": []}), encoding="utf-8")
    (tmp_path / "cache").mkdir()
    cache_file = tmp_path / "cache" / "toy_populations.json"
    ymtime = os.path.getmtime(tmp_path / "populations" / "toy.yaml")
    cache_file.write_text(json.dumps({"yaml_mtime": ymtime}), encoding="utf-8")   # нет "groups"
    r = populations.resolve("toy")
    assert r["mn9"] == ["1", "2"]


def test_resolve_invalidates_on_db_change(tmp_db, tmp_path, monkeypatch):
    """Кэш групп действителен, пока не изменилась версия набора в базе; если база стала новее
    кэша И версия сменилась (данные перезалиты) — resolve() пересчитывает, а не отдаёт старое."""
    monkeypatch.setattr(paths, "DB_PATH", tmp_db); monkeypatch.setattr(paths, "LAB", tmp_path); monkeypatch.setattr(paths, "CACHE", tmp_path / "cache")
    conn = _toy_db(tmp_db); conn.close()
    (tmp_path / "populations").mkdir()
    (tmp_path / "populations" / "toy.yaml").write_text(yaml.safe_dump({"dataset": "toy", "groups": {
        "mn9": {"where": {"cell_type": "MN9"}}}, "state": []}), encoding="utf-8")
    r1 = populations.resolve("toy")
    assert r1["mn9"] == ["1", "2"]
    cache_file = tmp_path / "cache" / "toy_populations.json"
    cache_mtime = os.path.getmtime(cache_file)

    # база перезалита с новой версией и третьим MN9 — mtime базы принудительно позже кэша
    conn = db.connect(tmp_db)
    db.register_dataset(conn, "toy", "", "2", "", {}, {})
    db.add_neurons(conn, "toy", [
        {"name": "1", "cell_type": "MN9", "cell_class": "motor", "side": "right", "extra": {}},
        {"name": "2", "cell_type": "MN9", "cell_class": "motor", "side": "left", "extra": {}},
        {"name": "5", "cell_type": "MN9", "cell_class": "motor", "side": "left", "extra": {}}])
    conn.commit()
    conn.close()
    os.utime(tmp_db, (cache_mtime + 10, cache_mtime + 10))

    r2 = populations.resolve("toy")
    assert r2["mn9"] == ["1", "2", "5"]     # пересчитано, а не старое значение из кэша


def test_worm_populations_match_groups(worm_raw):
    from brainlab.lab import groups
    if not paths.DB_PATH.exists():
        import pytest; pytest.skip("нет рабочей базы")
    r = populations.resolve("worm_cook2019")
    for k, v in groups.WORM.items():
        assert r[k] == v
