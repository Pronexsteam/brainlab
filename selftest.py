# -*- coding: utf-8 -*-
"""BrainLab self-test: technical (pytest), scientific (gate 1, habituation, gate 2),
honesty (run descriptions and the journal). The VERDICT is honest: it is computed over all
blocks at once, including gate 1 (closed by the 2026-09-14 decision, the worm is not being fixed)
— so the VERDICT can be red even when the fly science (gate 2) is green. Both gates are printed
on their own line, so this is visible without reading the code."""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from brainlab import paths  # noqa: E402

# Before this point, dataset_version could legitimately be absent from run.json: the field was
# introduced in the final wave of edits to plan 1-3 around 22:30 on 2026-09-14 (plan task 7, steps 4-6);
# the 22:31 cutoff is the controller's decision (a refinement of the earlier "22:00" estimate, based on the actual code).
# Runs whose folder name is newer than the cutoff must have dataset_version.
DATASET_VERSION_CUTOFF = "20260914-223100"


def _folder_stamp(name):
    m = re.match(r"^(\d{8}-\d{6})", name)
    return m.group(1) if m else None


def honesty(results_dir):
    """Honesty of runs and the atlas in results_dir. Returns (ok, report) — report is printed as-is."""
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
        "runs without run.json (rates.npz present)": no_run_json,
        "runs without code_hash": no_hash,
        "runs without dataset_version (newer than cutoff %s)" % DATASET_VERSION_CUTOFF: no_dataset_version,
        "runs with an invalid extra.extreme": bad_extreme,
        "atlas rows without run_id": atlas_bad,
    }
    ok = not (no_run_json or no_hash or no_dataset_version or bad_extreme or atlas_bad)
    return ok, report


def _gate2_dataset_present():
    """The flywire_630_shiu dataset is present if it is in the graph cache OR actually registered in
    the database (store.sqlite existing alone is not enough — the database can exist without this
    dataset in it, in which case graph.get() would crash instead of giving an honest red)."""
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
    print("== technical ==")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests", "-m", "not slow",
                         "--deselect", "tests/test_gate1.py",
                         "--deselect", "tests/test_runner.py::test_habituation_experiment",
                         "--deselect", "tests/test_gate2.py::test_gate2_passes"], cwd=ROOT)
    ok &= r.returncode == 0

    print("== scientific ==")
    from brainlab.lab.gates import gate1_worm_touch
    g1 = gate1_worm_touch.run(save=False)
    print("gate 1:", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in g1.items()}, ensure_ascii=False))
    gate1_ok = bool(g1["passed"])
    ok &= gate1_ok

    from brainlab.lab import runner
    hab_path = paths.LAB / "experiments" / "worm_tap_habituation.yaml"
    hab_spec = runner.load_experiment(hab_path)
    hab_r = runner.run_experiment(hab_path, save=False)
    hab_ev = runner.evaluate(hab_r, hab_spec)
    print("habituation:", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in hab_ev.items()}, ensure_ascii=False))
    ok &= hab_ev["valence"] > 0

    if not _gate2_dataset_present():
        print("gate 2: no reference dataset: python -m brainlab.store.fetch flywire_630_shiu "
              "&& python -m brainlab.store.loaders.fly_shiu630")
        gate2_ok = False
    else:
        from brainlab.lab.gates import gate2_fly_sugar
        g2 = gate2_fly_sugar.run(save=False, seeds=(0,))
        print("gate 2:", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in g2.items() if k != "run_ids"}, ensure_ascii=False))
        gate2_ok = bool(g2["passed"])
    ok &= gate2_ok

    print("gate 1 (worm, closed by the 2026-09-14 decision): %s" % ("green" if gate1_ok else "red"))
    print("gate 2 (fly): %s" % ("green" if gate2_ok else "red"))

    print("== honesty ==")
    ok_h, report = honesty(paths.RESULTS)
    for label, items in report.items():
        print("%s:" % label, items or "none")
    journal = paths.DOCS / "ЖУРНАЛ.md"
    has_journal = journal.exists() and journal.stat().st_size > 200
    print("journal:", "present" if has_journal else "missing")
    ok_h = ok_h and has_journal
    ok &= ok_h

    print("\nVERDICT:", "green" if ok else "RED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
