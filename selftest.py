# -*- coding: utf-8 -*-
"""Самопроверка BrainLab: техническая (pytest), научная (ворота 1, привыкание, ворота 2),
честности (описания прогонов и журнал). ИТОГ честный: считается по всем блокам сразу,
включая ворота 1 (закрыты решением от 2026-09-14, червя не чиним) — поэтому ИТОГ может
быть красным, даже когда наука по мухе (ворота 2) зелёная. Обе ворота печатаются отдельной
строкой, чтобы это было видно без разбора кода."""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from brainlab import paths  # noqa: E402

# До этого момента dataset_version мог отсутствовать в run.json легитимно: поле введено
# в финальной волне правок плана 1-3 около 22:30 2026-09-14 (задача 7 плана шаги 4-6);
# рубеж 22:31 — решение контролёра (уточнение прежней оценки "22:00" по факту кода).
# Прогоны с именем папки новее рубежа обязаны иметь dataset_version.
DATASET_VERSION_CUTOFF = "20260914-223100"


def _folder_stamp(name):
    m = re.match(r"^(\d{8}-\d{6})", name)
    return m.group(1) if m else None


def honesty(results_dir):
    """Честность прогонов и атласа в results_dir. Возвращает (ok, report) — report печатаем как есть."""
    results_dir = Path(results_dir)
    no_run_json, no_hash, no_dataset_version, bad_extreme, atlas_bad = [], [], [], [], []
    for folder in sorted(results_dir.glob("*")):
        if not folder.is_dir() or folder.name == "atlas":
            continue
        run_json = folder / "run.json"
        if (folder / "rates.npz").exists() and not run_json.exists():
            no_run_json.append(folder.name)
            continue
        if not run_json.exists():
            continue
        meta = json.loads(run_json.read_text(encoding="utf-8"))
        if not meta.get("code_hash"):
            no_hash.append(folder.name)
        if not meta.get("dataset_version"):
            stamp = _folder_stamp(folder.name)
            if stamp is None or stamp > DATASET_VERSION_CUTOFF:
                no_dataset_version.append(folder.name)
        extra = meta.get("extra") or {}
        if "extreme" in extra and not isinstance(extra["extreme"], bool):
            bad_extreme.append(folder.name)

    atlas_dir = results_dir / "atlas"
    if atlas_dir.exists():
        for f in sorted(atlas_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                atlas_bad.append(f.name)
                continue
            if not isinstance(data, dict):
                atlas_bad.append(f.name)
                continue
            for key, row in data.items():
                if not (isinstance(row, dict) and row.get("run_id")):
                    atlas_bad.append("%s:%s" % (f.name, key))

    report = {
        "прогонов без run.json (есть rates.npz)": no_run_json,
        "прогонов без code_hash": no_hash,
        "прогонов без dataset_version (новее рубежа %s)" % DATASET_VERSION_CUTOFF: no_dataset_version,
        "прогонов с некорректным extra.extreme": bad_extreme,
        "строк атласа без run_id": atlas_bad,
    }
    ok = not (no_run_json or no_hash or no_dataset_version or bad_extreme or atlas_bad)
    return ok, report


def _gate2_dataset_present():
    """Набор flywire_630_shiu есть, если он в кэше графа ИЛИ реально зарегистрирован в базе
    (голого наличия store.sqlite мало — база может быть, а этого набора в ней нет; тогда
    graph.get() упадёт вместо честного красного)."""
    if (paths.CACHE / "flywire_630_shiu.npz").exists():
        return True
    if not paths.DB_PATH.exists():
        return False
    from brainlab.store import db
    conn = db.connect()
    try:
        return db.dataset_info(conn, "flywire_630_shiu") is not None
    finally:
        conn.close()


def main():
    ok = True
    print("== техническая ==")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests", "-m", "not slow",
                         "--deselect", "tests/test_gate1.py",
                         "--deselect", "tests/test_runner.py::test_habituation_experiment",
                         "--deselect", "tests/test_gate2.py::test_gate2_passes"], cwd=ROOT)
    ok &= r.returncode == 0

    print("== научная ==")
    from brainlab.lab.gates import gate1_worm_touch
    g1 = gate1_worm_touch.run(save=False)
    print("ворота 1:", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in g1.items()}, ensure_ascii=False))
    gate1_ok = bool(g1["passed"])
    ok &= gate1_ok

    from brainlab.lab import runner
    hab_path = paths.LAB / "experiments" / "worm_tap_habituation.yaml"
    hab_spec = runner.load_experiment(hab_path)
    hab_r = runner.run_experiment(hab_path, save=False)
    hab_ev = runner.evaluate(hab_r, hab_spec)
    print("привыкание:", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in hab_ev.items()}, ensure_ascii=False))
    ok &= hab_ev["valence"] > 0

    if not _gate2_dataset_present():
        print("ворота 2: нет набора эталона: python -m brainlab.store.fetch flywire_630_shiu "
              "&& python -m brainlab.store.loaders.fly_shiu630")
        gate2_ok = False
    else:
        from brainlab.lab.gates import gate2_fly_sugar
        g2 = gate2_fly_sugar.run(save=False, seeds=(0,))
        print("ворота 2:", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in g2.items() if k != "run_ids"}, ensure_ascii=False))
        gate2_ok = bool(g2["passed"])
    ok &= gate2_ok

    print("ворота 1 (червь, закрыт решением от 2026-09-14): %s" % ("зелёный" if gate1_ok else "красный"))
    print("ворота 2 (муха): %s" % ("зелёный" if gate2_ok else "красный"))

    print("== честности ==")
    ok_h, report = honesty(paths.RESULTS)
    for label, items in report.items():
        print("%s:" % label, items or "нет")
    journal = paths.DOCS / "ЖУРНАЛ.md"
    has_journal = journal.exists() and journal.stat().st_size > 200
    print("журнал:", "есть" if has_journal else "нет")
    ok_h = ok_h and has_journal
    ok &= ok_h

    print("\nИТОГ:", "зелёный" if ok else "КРАСНЫЙ")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
