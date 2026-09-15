"""Три полных скана взрослой мухи в едином формате лаборатории Lee (открытый бакет, см. fetch.FILES):
fafb_783 (FlyWire v783, самка, мозг), banc_888 (самка, мозг + брюшной узел), malecns_09 (самец, ЦНС).

Допущения:
- знак ребра по предсказанному медиатору пресинаптической клетки (дизайн §4): ацетилхолин +1,
  ГАМК −1, глутамат −1 (у мухи GluCl), гистамин −1 (у мухи фоторецепторы → ламина через
  гистамин-хлорные каналы HisCl, тормозный), моноамины (дофамин, серотонин, октопамин) и
  тирамин — MONOAMINE_SIGN (см. ниже, выверено по эталону Shiu только для дофамина/серотонина/
  октопамина; тирамин к тому же классу отнесён по решению постановщика, отдельно по эталону не
  проверялся), "unclear"/пусто/что угодно ещё не из списка — 0 (ребро остаётся в базе с нулевым
  знаком, в W_chem не попадает);
- порог хранения min_count = 5 синапсов на пару (как «connections» у Codex); эталонный набор
  Shiu хранит всё, поэтому сравнивать числа рёбер между ними нельзя напрямую;
- cell_class в базе = super_class meta (верхний класс); подробные cell_class/cell_sub_class — в extra;
- skipped_edges_unknown_id считает только рёбра, прошедшие фильтр min_count (у кого pre или post
  не нашлись в meta) — это нижняя оценка доли «потерянной» связности: рёбра ниже min_count в этот
  счётчик не попадают, даже если их pre/post тоже отсутствуют в meta.

Знак моноаминов (шаг 3б, выверка по эталону Shiu 2024, flywire_630_shiu, id v630 = v783 у
выживших клеток): в meta FAFB v783 клеток с neurotransmitter_predicted в {dopamine, serotonin,
octopamine} — 8853; из них id нашлись как Presynaptic_ID в parquet эталона (2023_03_23_connectivity
_630_final.parquet) у 7373 клеток, дающих 656999 исходящих рёбер. Знак по столбцу Excitatory:
dopamine 570757 pos / 5 neg, serotonin 78202 pos / 125 neg, octopamine 7859 pos / 51 neg;
суммарно 656818 pos / 181 neg, доля положительных 0.9997 (по каждому медиатору отдельно —
0.9999 / 0.9984 / 0.9936). У эталона моноамины практически однозначно положительные (редкие
минусы — шум предсказания на единичных клетках), поэтому MONOAMINE_SIGN = 1. Это выверка
допущения по чужим данным, не подгонка под ворота (моноамины не входят в ворота 1/2).

Список рёбер читается батчами лениво через pyarrow (у MaleCNS 3,2 ГБ): открытый файл/память
memory_map закрываются, как только генератор рёбер исчерпан или прерван исключением.
"""
import hashlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as ft

from .. import db

SPECS = {
    "fafb_783": {"meta": "fafb_783_meta.feather", "edges": "fafb_783_simple_edgelist.feather", "id_col": "fafb_783_id",
                 "version": "fafb_783", "source": "FlyWire/FAFB v783 via lee-lab bucket", "license": "CC-BY 4.0"},
    "banc_888": {"meta": "banc_888_meta.feather", "edges": "banc_888_edgelist_simple_v3.feather", "id_col": "banc_888_id",
                 "version": "banc_888 v3", "source": "BANC v888 via lee-lab bucket", "license": "CC-BY 4.0"},
    "malecns_09": {"meta": "malecns_09_meta.feather", "edges": "malecns_09_simple_edgelist.feather", "id_col": "malecns_09_id",
                   "version": "malecns_09", "source": "MaleCNS v0.9 via lee-lab bucket", "license": "CC-BY 4.0"},
}
MONOAMINE_SIGN = 1            # выверено в задаче 4, шаг 3б по эталону Shiu (см. докстринг выше)
SIGN = {"acetylcholine": 1, "gaba": -1, "glutamate": -1, "histamine": -1}
MONOAMINES = ("dopamine", "serotonin", "octopamine", "tyramine")   # -> MONOAMINE_SIGN (см. допущения)
SIGN_RULE = ("presynaptic neurotransmitter_predicted: ach +1, gaba -1, glutamate -1, histamine -1, "
             "dopamine/serotonin/octopamine/tyramine MONOAMINE_SIGN, unclear/unknown/empty 0")
EXTRA_COLS = ["cell_class", "cell_sub_class", "cell_function", "cell_function_detailed", "flow",
              "neurotransmitter_score", "fafb_cell_type", "body_part_sensory", "body_part_effector"]
_NA_STRINGS = ("", "nan", "none", "<na>")   # регистронезависимо: сентинели пропуска в строковых столбцах Arrow


def _s(v):
    if v is None:
        return ""
    if isinstance(v, float) and np.isnan(v):
        return ""
    s = str(v)
    return "" if s.strip().lower() in _NA_STRINGS else s


def rule_hash(min_count):
    """Короткий хеш правила знаков + порога: меняется вместе со сменой SIGN_RULE/min_count/
    MONOAMINE_SIGN, поэтому смена правила меняет version набора в базе (задача 1 финальной волны)."""
    h = hashlib.sha256((SIGN_RULE + str(min_count) + str(MONOAMINE_SIGN)).encode("utf-8")).hexdigest()
    return h[:8]


def version_string(ds, min_count):
    return "%s (lee-lab compiled_data, 2026; rule %s)" % (ds, rule_hash(min_count))


def _sign(tr):
    if tr in SIGN:
        return SIGN[tr]
    if tr in MONOAMINES:
        return MONOAMINE_SIGN
    return 0


def _edge_batches(edge_f):
    """Ленивый генератор батчей рёбер: memory_map/reader открываются один раз и закрываются
    (finally), как только источник исчерпан или прерван исключением. Пробуем IPC file-формат
    (у всех трёх наборов Lee-lab на 2026-09-14 — он), при ArrowInvalid — IPC stream (батчи там
    читаются по одному без произвольного доступа, тоже лениво)."""
    mm = pa.memory_map(str(edge_f), "r")
    try:
        reader = pa.ipc.open_file(mm)
        is_file = True
    except pa.ArrowInvalid:
        mm.seek(0)
        reader = pa.ipc.open_stream(mm)
        is_file = False
    try:
        yield reader.schema
        if is_file:
            for i in range(reader.num_record_batches):
                yield reader.get_batch(i)
        else:
            for batch in reader:
                yield batch
    finally:
        mm.close()


def load(conn, raw_dir, dataset_id, min_count=5):
    spec = SPECS[dataset_id]
    raw_dir = Path(raw_dir)
    meta_f, edge_f = raw_dir / spec["meta"], raw_dir / spec["edges"]
    meta = ft.read_table(meta_f).to_pandas()
    idc = spec["id_col"]
    if idc not in meta.columns:
        raise ValueError("в %s нет столбца %s" % (meta_f.name, idc))
    rows, sign_of = [], {}
    for r in meta.itertuples(index=False):
        d = r._asdict()
        name, tr = _s(d[idc]), _s(d.get("neurotransmitter_predicted"))
        extra = {k: _s(d[k]) for k in EXTRA_COLS if k in d and _s(d[k])}
        rows.append({"name": name, "cell_type": _s(d.get("cell_type")), "cell_class": _s(d.get("super_class")),
                     "transmitter": tr, "side": _s(d.get("side")), "region": _s(d.get("region")), "extra": extra})
        sign_of[name] = _sign(tr)
    files = {p.name: db.sha256_file(p) for p in (meta_f, edge_f)}
    src = _edge_batches(edge_f)
    schema = next(src)
    need = {"pre", "post", "count"}
    if not need <= set(schema.names):
        raise ValueError("в %s нет столбцов %s (есть: %s)" % (edge_f.name, sorted(need - set(schema.names)), schema.names))
    stats = {"edges": 0, "skipped_edges_unknown_id": 0}

    def gen():
        for batch in src:
            b = batch.select(["pre", "post", "count"])
            pre, post, cnt = b.column(0).to_pylist(), b.column(1).to_pylist(), b.column(2).to_numpy()
            for a, c, k in zip(pre, post, cnt):
                if k < min_count:
                    continue
                a, c = str(a), str(c)
                if a not in sign_of or c not in sign_of:
                    stats["skipped_edges_unknown_id"] += 1
                    continue
                stats["edges"] += 1
                yield {"pre": a, "post": c, "kind": "chemical", "count": float(k), "sign": sign_of[a]}

    try:
        # одна транзакция на загрузчик (см. worm_cook2019.load): упавший на середине генератор
        # рёбер (в т.ч. в _edge_batches) откатывает и register_dataset, и add_neurons — набора в базе нет
        db.register_dataset(conn, dataset_id, spec["source"], version_string(spec["version"], min_count), spec["license"], files,
                            {"sign_rule": SIGN_RULE, "min_count": min_count, "monoamine_sign": MONOAMINE_SIGN, "id_col": idc})
        db.add_neurons(conn, dataset_id, rows)
        db.add_edges(conn, dataset_id, gen())
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    by_tr = {}
    for r in rows:
        by_tr[r["transmitter"] or "none"] = by_tr.get(r["transmitter"] or "none", 0) + 1
    return {"neurons": len(rows), "edges": stats["edges"], "by_transmitter": by_tr,
            "skipped_edges_unknown_id": stats["skipped_edges_unknown_id"]}


if __name__ == "__main__":
    import sys
    from .. import fetch
    c = db.connect()
    for ds in sys.argv[1:] or list(SPECS):
        print(ds, load(c, fetch.raw_dir(ds), ds))
    c.close()
