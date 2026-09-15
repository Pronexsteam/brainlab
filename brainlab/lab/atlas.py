"""Атлас устойчивости (дизайн §14а): один и тот же опыт на нескольких наборах → одна таблица.
Файлы атласа не перезаписываются: новая версия получает суффикс -1, -2, …"""
import json
import time

import yaml

from .. import paths
from ..store import db
from . import populations, runner

_META_KEYS = ("run_id", "valence", "flags", "empty_groups", "version", "min_count", "sign_rule")


def _free(base, ext):
    f = base.with_suffix(ext)
    k = 0
    while f.exists():
        k += 1
        f = base.with_name("%s-%d%s" % (base.name, k, ext))
    return f


def _dataset_meta(ds):
    """version/min_count/sign_rule набора из db.dataset_info(...)["params"] — задача 4
    финальной волны: атлас идёт на графах с порогом min_count, модель откалибрована на всём графе."""
    conn = db.connect()
    try:
        info = db.dataset_info(conn, ds)
    finally:
        conn.close()
    version = (info or {}).get("version", "")
    params = json.loads((info or {}).get("params") or "{}")
    return version, params.get("min_count", "-"), params.get("sign_rule", "-")


def run(name, datasets, save=True):
    out = {}
    for ds in datasets:
        p = paths.LAB / "experiments" / ("%s_%s.yaml" % (name, ds))
        spec = yaml.safe_load(p.read_text(encoding="utf-8"))
        if not ds.startswith(spec["dataset"]):
            # токен файла начинается с dataset: banc_888_norm → dataset banc_888 (тот же набор,
            # нормированные входы); для обычных наборов это по-прежнему полное совпадение
            raise ValueError("%s: dataset в yaml (%r) не совпадает с токеном файла (%r)" % (p, spec["dataset"], ds))
        r = runner.run_experiment(p, save=save)
        version, min_count, sign_rule = _dataset_meta(spec["dataset"])
        if r.extra.get("graph", {}).get("normalize"):
            version += " norm:" + r.extra["graph"]["normalize"]
        row = {"run_id": r.id, "valence": r.valence, "flags": r.flags, "empty_groups": r.extra.get("empty_groups", []),
               "version": version, "min_count": min_count, "sign_rule": sign_rule}
        for k, m in spec.get("measure", {}).items():
            names = populations.names(spec["dataset"], m["group"])
            row[k] = float(r.hz(names, *m["window"])) if names else 0.0
        out[ds] = row
    folder = paths.RESULTS / "atlas"
    folder.mkdir(parents=True, exist_ok=True)
    base = folder / name
    keys = sorted({k for row in out.values() for k in row if k not in _META_KEYS})
    lines = ["# Атлас: %s (%s)\n" % (name, time.strftime("%Y-%m-%d %H:%M")),
             "| набор | версия | min_count | правило знаков | " + " | ".join(keys) + " | оценка | флаги | пустые группы | прогон |",
             "|" + " --- |" * (len(keys) + 8)]
    for ds, row in out.items():
        lines.append("| %s | %s | %s | %s | %s | %+d | %s | %s | %s |" % (
                     ds, row["version"] or "-", row["min_count"], row["sign_rule"] or "-",
                     " | ".join("%.1f" % row[k] for k in keys), row["valence"],
                     ",".join(k for k, v in row["flags"].items() if v is True) or "-", ",".join(row["empty_groups"]) or "-", row["run_id"] or "-"))
    lines.append("\nмодель откалибрована воротами 2 на графе со всеми синапсами (эталон Shiu); атлас идёт "
                  "на графах с порогом min_count=5; контрольный прогон fafb_783 при min_count=1 — задача "
                  "следующего плана.\n")
    _free(base, ".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _free(base, ".json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    import sys
    print(json.dumps(run(sys.argv[1], sys.argv[2:] or ["fafb_783", "banc_888", "malecns_09"]), ensure_ascii=False, indent=1))
