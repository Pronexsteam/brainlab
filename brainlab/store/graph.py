"""Граф набора: матрицы связей для симулятора и подписи клеток.

W_chem[post, pre] = знак * число синапсов: активность = W_chem @ r.
W_gap симметрична, вес = число контактов. Кэш — npz рядом с базой, пересобирается,
если база новее кэша.

Нормировка входов по типу клетки (get(ds, normalize=ref), normalize_inputs) — перенос модели
Shiu (абсолютный вес мВ/синапс, подобран под плотность FAFB) на скан с другой плотностью синапсов.
"""
import json
import os
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .. import paths
from . import db


@dataclass
class Graph:
    dataset: str
    names: list
    cell_class: list
    transmitter: list
    W_chem: sp.csr_matrix
    W_gap: sp.csr_matrix
    version: str = ""
    files: dict = None
    params: dict = None    # datasets.params набора (sign_rule/min_count/...), задача 1 финальной волны

    def __post_init__(self):
        self.files = self.files or {}
        self.params = self.params or {}
        self.index = {n: i for i, n in enumerate(self.names)}

    @property
    def n(self):
        return len(self.names)

    def idx(self, names):
        return np.array([self.index[n] for n in names], dtype=np.int64)


def build(conn, dataset_id):
    info = db.dataset_info(conn, dataset_id)
    if info is None:
        raise KeyError("набор %r не зарегистрирован в базе (dataset_info пуст) — сначала загрузи его загрузчиком" % dataset_id)
    ns = db.neurons(conn, dataset_id)
    names = [n["name"] for n in ns]
    index = {n: i for i, n in enumerate(names)}
    n = len(names)
    rows = conn.execute("SELECT pre, post, kind, count, sign FROM edges WHERE dataset = ?", (dataset_id,)).fetchall()
    pre = np.fromiter((index.get(r[0], -1) for r in rows), dtype=np.int64, count=len(rows))
    post = np.fromiter((index.get(r[1], -1) for r in rows), dtype=np.int64, count=len(rows))
    elec = np.fromiter((r[2] == "electrical" for r in rows), dtype=bool, count=len(rows))
    cnt = np.fromiter((r[3] for r in rows), dtype=np.float32, count=len(rows))
    sgn = np.fromiter((r[4] for r in rows), dtype=np.float32, count=len(rows))
    ok = (pre >= 0) & (post >= 0)
    c = ok & ~elec
    W_chem = sp.csr_matrix((sgn[c] * cnt[c], (post[c], pre[c])), shape=(n, n), dtype=np.float32)
    e = ok & elec
    W_gap = sp.csr_matrix((cnt[e], (post[e], pre[e])), shape=(n, n), dtype=np.float32)
    W_gap.sum_duplicates()
    # одна запись на строку csv (b=post, a=pre); большинство контактов и так перечислены в обе
    # стороны, но 26 из 2698 — только в одну, поэтому симметризуем через максимум (не через
    # (W+Wᵀ)/2, иначе двусторонние записи снова считались бы дважды).
    W_gap = W_gap.maximum(W_gap.T).tocsr()
    files = json.loads(info.get("files") or "{}")
    params = json.loads(info.get("params") or "{}")
    W_chem.eliminate_zeros()    # рёбра со знаком 0 (неизвестный медиатор) не должны быть явными нулями в nnz
    return Graph(dataset_id, names, [x["cell_class"] for x in ns], [x["transmitter"] for x in ns], W_chem, W_gap,
                version=info.get("version") or "", files=files, params=params)


FACTOR_MIN, FACTOR_MAX = 0.2, 5.0     # границы множителя нормировки (не крутить под ответ MN9)
NORM_RULE = "per-cell input scaled to median input of same fafb_cell_type in ref"


def _cache_file(dataset_id, normalize=None):
    return paths.CACHE / ("%s.npz" % dataset_id if not normalize else "%s__norm-%s.npz" % (dataset_id, normalize))


def save_cache(g, normalize=None):
    paths.CACHE.mkdir(parents=True, exist_ok=True)
    meta = json.dumps({"names": g.names, "cell_class": g.cell_class, "transmitter": g.transmitter,
                       "version": g.version, "files": g.files, "params": g.params}, ensure_ascii=False)
    c, gp = g.W_chem.tocoo(), g.W_gap.tocoo()
    np.savez_compressed(_cache_file(g.dataset, normalize), meta=np.array(meta), n=g.n,
                        c_row=c.row, c_col=c.col, c_val=c.data, g_row=gp.row, g_col=gp.col, g_val=gp.data)


def load_cache(dataset_id, normalize=None):
    f = _cache_file(dataset_id, normalize)
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    n = int(z["n"])
    W_chem = sp.csr_matrix((z["c_val"], (z["c_row"], z["c_col"])), shape=(n, n), dtype=np.float32)
    W_gap = sp.csr_matrix((z["g_val"], (z["g_row"], z["g_col"])), shape=(n, n), dtype=np.float32)
    return Graph(dataset_id, meta["names"], meta["cell_class"], meta["transmitter"], W_chem, W_gap,
                version=meta.get("version", ""), files=meta.get("files", {}), params=meta.get("params", {}))


def _cell_types(conn, dataset_id, names):
    """Тип клетки для нормировки, в порядке names: extra["fafb_cell_type"] (BANC/MaleCNS), если
    пусто — cell_type (FAFB). conn=None — открыть свою."""
    owns_conn = conn is None
    conn = conn or db.connect()
    try:
        by_name = {r["name"]: (r["extra"].get("fafb_cell_type") or r["cell_type"] or "") for r in db.neurons(conn, dataset_id)}
    finally:
        if owns_conn:
            conn.close()
    return [by_name.get(n, "") for n in names]


def _in_abs(W):
    """Суммарный вход клетки: Σ|W_chem[j, :]| — число синапсов до знака; рёбра со знаком 0
    вне W_chem (eliminate_zeros в build) и не считаются."""
    return np.asarray(abs(W).sum(axis=1)).ravel().astype(np.float64)


def normalize_inputs(g, g_ref, types, ref_types, factor_min=FACTOR_MIN, factor_max=FACTOR_MAX):
    """Правило (допущение): привести суммарный вход каждой клетки g к медиане входа клеток того же
    типа в опорном графе g_ref. in[j] = Σ|W_chem[j,:]| (по abs, до знака); target[t] = медиана in_ref
    по клеткам ref типа t; для клетки j типа t (types[j], у BANC/MaleCNS это fafb_cell_type, если
    пусто — cell_type): если t есть в target и in[j] > 0, factor[j] = clip(target[t]/in[j],
    factor_min, factor_max), иначе 1.0. Строка j W_chem (все входы клетки) умножается на factor[j];
    W_gap не трогается. Возвращает новый Graph (исходный не меняется): dataset тот же,
    version + " norm:" + ref, params["normalize"] со статистикой."""
    if len(types) != g.n or len(ref_types) != g_ref.n:
        raise ValueError("список типов не совпадает по длине с графом")
    in_ref = _in_abs(g_ref.W_chem)
    by_type = {}
    for t, v in zip(ref_types, in_ref):
        if t:
            by_type.setdefault(t, []).append(v)
    target = {t: float(np.median(v)) for t, v in by_type.items()}
    in_ds = _in_abs(g.W_chem)
    factor = np.ones(g.n, dtype=np.float64)
    scaled = np.zeros(g.n, dtype=bool)
    for j, t in enumerate(types):
        if t and t in target and in_ds[j] > 0:
            factor[j] = min(max(target[t] / in_ds[j], factor_min), factor_max)
            scaled[j] = True
    W = sp.csr_matrix(sp.diags(factor.astype(np.float32)) @ g.W_chem, dtype=np.float32)
    W.eliminate_zeros()
    params = dict(g.params)
    q = np.quantile(factor[scaled], [0.05, 0.25, 0.5, 0.75, 0.95]) if scaled.any() else np.ones(5)
    params["normalize"] = {"ref": g_ref.dataset, "rule": NORM_RULE, "factor_min": factor_min, "factor_max": factor_max,
                           "cells_scaled": int(scaled.sum()), "cells_unscaled": int((~scaled).sum()),
                           "factor_median": float(np.median(factor)),
                           "factor_scaled_quantiles_5_25_50_75_95": [round(float(x), 4) for x in q],
                           "at_min": int((scaled & (factor <= factor_min)).sum()),
                           "at_max": int((scaled & (factor >= factor_max)).sum()),
                           "types_in_ref": len(target)}
    return Graph(g.dataset, list(g.names), list(g.cell_class), list(g.transmitter), W, g.W_gap.copy(),
                 version="%s norm:%s" % (g.version, g_ref.dataset), files=dict(g.files), params=params)


def _cache_fresh(f):
    # кэш есть и (базы нет ИЛИ кэш не старее базы) — используем кэш, не пересобираем
    # пустой граф, если рабочая база вдруг отсутствует.
    return f.exists() and (not paths.DB_PATH.exists() or os.path.getmtime(f) >= os.path.getmtime(paths.DB_PATH))


def get(dataset_id, conn=None, normalize=None):
    """Граф набора из кэша или базы. normalize=<ref> — граф с входами, нормированными на опорный
    набор ref (normalize_inputs), кэш data/cache/<ds>__norm-<ref>.npz."""
    if normalize:
        if _cache_fresh(_cache_file(dataset_id, normalize)):
            g = load_cache(dataset_id, normalize)
            if g is not None:
                return g
        g, g_ref = get(dataset_id, conn), get(normalize, conn)
        gn = normalize_inputs(g, g_ref, _cell_types(conn, dataset_id, g.names), _cell_types(conn, normalize, g_ref.names))
        save_cache(gn, normalize)
        return gn
    if _cache_fresh(_cache_file(dataset_id)):
        g = load_cache(dataset_id)
        if g is not None:
            return g
    owns_conn = conn is None
    conn = conn or db.connect()
    try:
        g = build(conn, dataset_id)
    finally:
        if owns_conn:
            conn.close()
    save_cache(g)
    return g
