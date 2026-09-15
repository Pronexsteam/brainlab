"""Сырьё наборов: откуда качать и куда класть. data/<набор>/raw/<имя>. Готовые файлы не перекачиваются.

Адреса — открытый бакет лаборатории Lee (без входа) и репозиторий эталонной модели Shiu 2024.
При смене адресов правится только таблица FILES (дизайн §16)."""
import shutil
import urllib.request
from pathlib import Path

from .. import paths
from .db import sha256_file

_LEE = "https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/"
_SHIU = "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/"

_C302 = "https://raw.githubusercontent.com/openworm/c302/master/c302/data/"

FILES = {
    "worm": [(_C302 + "herm_full_edgelist.csv", "herm_full_edgelist.csv"),
             (_C302 + "aconnectome_white_1986_whole.csv", "aconnectome_white_1986_whole.csv"),
             (_C302 + "Bentley_et_al_2016_expression.csv", "Bentley_et_al_2016_expression.csv")],
    "fafb_783": [(_LEE + "fafb_783/fafb_783_meta.feather", "fafb_783_meta.feather"),
                 (_LEE + "fafb_783/fafb_783_simple_edgelist.feather", "fafb_783_simple_edgelist.feather")],
    "banc_888": [(_LEE + "banc_888/banc_888_meta.feather", "banc_888_meta.feather"),
                 (_LEE + "banc_888/banc_888_edgelist_simple_v3.feather", "banc_888_edgelist_simple_v3.feather")],
    "malecns_09": [(_LEE + "malecns_09/malecns_09_meta.feather", "malecns_09_meta.feather"),
                   (_LEE + "malecns_09/malecns_09_simple_edgelist.feather", "malecns_09_simple_edgelist.feather")],
    "flywire_630_shiu": [(_SHIU + "2023_03_23_completeness_630_final.csv", "2023_03_23_completeness_630_final.csv"),
                         (_SHIU + "2023_03_23_connectivity_630_final.parquet", "2023_03_23_connectivity_630_final.parquet"),
                         (_SHIU + "results/example/sugarR.parquet", "sugarR.parquet"),
                         (_SHIU + "model.py", "model.py"), (_SHIU + "utils.py", "utils.py"),
                         (_SHIU + "example.ipynb", "example.ipynb"), (_SHIU + "sez_neurons.pickle", "sez_neurons.pickle")],
}


# первичные таблицы синапсов (2 и 20 ГБ) — только для lab/analysis/banc_native_synapses.py
FILES_EXTRA = {
    "fafb_783": [(_LEE + "fafb_783/fafb_783_synapses.parquet", "fafb_783_synapses.parquet")],
    "banc_888": [(_LEE + "banc_888/banc_888_synapses_v3_enriched.parquet", "banc_888_synapses_v3_enriched.parquet")],
}


def raw_dir(dataset_id):
    if dataset_id == "worm":
        return paths.DATA / "worm"          # исторически без подпапки raw (загрузчик червя читает отсюда)
    return paths.DATA / dataset_id / "raw"


def _download(url, dest):
    dest = Path(dest)
    part = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(part, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    part.replace(dest)


def fetch(dataset_id, only=None, download=_download, extra=False):
    if dataset_id not in FILES:
        raise KeyError("набор %r не в таблице FILES" % dataset_id)
    d = raw_dir(dataset_id)
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for url, name in FILES[dataset_id] + (FILES_EXTRA.get(dataset_id, []) if extra else []):
        if only and name not in only:
            continue
        dest = d / name
        if not dest.exists():
            print("качаю", url)
            download(url, dest)
        out.append(dest)
    return out


def manifest(dataset_id):
    d = raw_dir(dataset_id)
    return {p.name: sha256_file(p) for p in sorted(d.glob("*")) if p.is_file() and not p.name.endswith(".part")}


if __name__ == "__main__":
    import sys
    args = [a for a in sys.argv[1:] if a != "--extra"]
    for ds in args or FILES:
        for p in fetch(ds, extra="--extra" in sys.argv):
            print(p, p.stat().st_size)
