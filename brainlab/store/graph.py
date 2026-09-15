"""A dataset's graph: connectivity matrices for the simulator and cell labels.

W_chem[post, pre] = sign * synapse count: activity = W_chem @ r.
W_gap is symmetric, weight = number of contacts. Cache is an npz next to the database, rebuilt
if the database is newer than the cache.

Input normalization by cell type (get(ds, normalize=ref), normalize_inputs) — carries Shiu's
model (an absolute mV/synapse weight, tuned to FAFB's synapse density) over to a scan with a
different synapse density.
"""
import hashlib
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
    params: dict = None    # the dataset's datasets.params (sign_rule/min_count/...), task 1 of the final wave

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
        raise KeyError("dataset %r is not registered in the database (dataset_info is empty) — load it with a loader first" % dataset_id)
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
    # one entry per csv row (b=post, a=pre); most contacts are already listed both ways,
    # but 26 of 2698 are listed only one way, so we symmetrize by taking the maximum (not
    # (W+Wᵀ)/2, which would double-count the two-way entries again).
    W_gap = W_gap.maximum(W_gap.T).tocsr()
    files = json.loads(info.get("files") or "{}")
    params = json.loads(info.get("params") or "{}")
    W_chem.eliminate_zeros()    # edges with sign 0 (unknown transmitter) must not be explicit zeros in nnz
    return Graph(dataset_id, names, [x["cell_class"] for x in ns], [x["transmitter"] for x in ns], W_chem, W_gap,
                version=info.get("version") or "", files=files, params=params)


FACTOR_MIN, FACTOR_MAX = 0.2, 5.0     # bounds of the normalization factor (do not tune this to fit the MN9 response)
NORM_RULE = "per-cell input scaled to median input (over ref cells with input > 0) of same fafb_cell_type in ref"
MEDIAN_OVER = "positive"              # the type target is the median over reference cells with in_ref > 0
AUTO_PREFIX = "auto:"                 # automatic type labels in BANC/MaleCNS (auto:L5 for the FAFB type L5)


def _norm_options(opts):
    """Normalization options in canonical form: strip_auto (bool), regions (sorted list or None),
    median_over (fixed, so a canonical dict passes through unchanged). Unknown keys are an error, so a typo
    in the YAML does not silently run the default."""
    opts = dict(opts or {})
    unknown = set(opts) - {"strip_auto", "regions", "median_over"}
    if unknown:
        raise ValueError("unknown normalization options %s (known: strip_auto, regions)" % sorted(unknown))
    regions = opts.get("regions")
    return {"strip_auto": bool(opts.get("strip_auto", False)),
            "regions": sorted(str(r) for r in regions) if regions else None, "median_over": MEDIAN_OVER}


def _cache_file(dataset_id, normalize=None, norm_opts=None):
    """data/cache/<ds>.npz; a normalized graph — <ds>__norm-<ref>-<hash>.npz, the hash covering the
    canonical options (strip_auto, regions, median_over), so a change of rule never reuses an older cache."""
    if not normalize:
        return paths.CACHE / ("%s.npz" % dataset_id)
    h = hashlib.sha256(json.dumps(_norm_options(norm_opts), sort_keys=True).encode("utf-8")).hexdigest()[:8]
    return paths.CACHE / ("%s__norm-%s-%s.npz" % (dataset_id, normalize, h))


def save_cache(g, normalize=None, norm_opts=None):
    paths.CACHE.mkdir(parents=True, exist_ok=True)
    meta = json.dumps({"names": g.names, "cell_class": g.cell_class, "transmitter": g.transmitter,
                       "version": g.version, "files": g.files, "params": g.params}, ensure_ascii=False)
    c, gp = g.W_chem.tocoo(), g.W_gap.tocoo()
    np.savez_compressed(_cache_file(g.dataset, normalize, norm_opts), meta=np.array(meta), n=g.n,
                        c_row=c.row, c_col=c.col, c_val=c.data, g_row=gp.row, g_col=gp.col, g_val=gp.data)


def load_cache(dataset_id, normalize=None, norm_opts=None):
    f = _cache_file(dataset_id, normalize, norm_opts)
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
    """Cell type for normalization, in the order of names: extra["fafb_cell_type"] (BANC/MaleCNS), if
    empty — cell_type (FAFB). conn=None — open one's own."""
    owns_conn = conn is None
    conn = conn or db.connect()
    try:
        by_name = {r["name"]: (r["extra"].get("fafb_cell_type") or r["cell_type"] or "") for r in db.neurons(conn, dataset_id)}
    finally:
        if owns_conn:
            conn.close()
    return [by_name.get(n, "") for n in names]


def _cell_regions(conn, dataset_id, names):
    """The region column of the neurons table, in the order of names (central_brain, optic_lobe,
    ventral_nerve_cord, ...; "" when unknown). conn=None — open one's own."""
    owns_conn = conn is None
    conn = conn or db.connect()
    try:
        by_name = {r["name"]: (r["region"] or "") for r in db.neurons(conn, dataset_id)}
    finally:
        if owns_conn:
            conn.close()
    return [by_name.get(n, "") for n in names]


def _in_abs(W):
    """Total input of a cell: Σ|W_chem[j, :]| — synapse count before applying sign; edges with sign 0
    are outside W_chem (eliminate_zeros in build) and are not counted."""
    return np.asarray(abs(W).sum(axis=1)).ravel().astype(np.float64)


def normalize_inputs(g, g_ref, types, ref_types, factor_min=FACTOR_MIN, factor_max=FACTOR_MAX,
                     strip_auto=False, regions=None, cell_regions=None):
    """Rule (assumption): scale each cell's total input in g to the median input of cells of the same
    type in the reference graph g_ref. in[j] = Σ|W_chem[j,:]| (by abs, before sign); target[t] = median
    of in_ref over ref cells of type t with in_ref > 0 (cells with no inputs at all are unreconstructed,
    not weak, and would pull the target down; params["median_over"] = "positive"); for a cell j of type t
    (types[j]; for BANC/MaleCNS this is fafb_cell_type, if empty — cell_type): if t is in target and
    in[j] > 0, factor[j] = clip(target[t]/in[j], factor_min, factor_max), otherwise 1.0.
    strip_auto=True: a type "auto:X" in g is matched as "X" (automatic labels in BANC/MaleCNS);
    params["auto_matched"] counts the cells this gave a target to. regions=[...]: only cells whose
    region (cell_regions[j], the neurons.region column) is in the list are scaled, the rest keep 1.0;
    params["cells_outside_regions"] counts them. Row j of W_chem (all of the cell's inputs) is
    multiplied by factor[j]; W_gap is left untouched. Returns a new Graph (the original is unchanged):
    same dataset, version + " norm:" + ref, params["normalize"] with the options and statistics."""
    if len(types) != g.n or len(ref_types) != g_ref.n:
        raise ValueError("the list of types does not match the graph's length")
    if regions is not None:
        if cell_regions is None or len(cell_regions) != g.n:
            raise ValueError("regions=... needs cell_regions, one region per cell of the graph")
        allowed = set(regions)
        in_region = np.array([r in allowed for r in cell_regions], dtype=bool)
    else:
        in_region = np.ones(g.n, dtype=bool)
    in_ref = _in_abs(g_ref.W_chem)
    by_type = {}
    for t, v in zip(ref_types, in_ref):
        if t and v > 0:
            by_type.setdefault(t, []).append(v)
    target = {t: float(np.median(v)) for t, v in by_type.items()}
    in_ds = _in_abs(g.W_chem)
    factor = np.ones(g.n, dtype=np.float64)
    scaled = np.zeros(g.n, dtype=bool)
    auto_matched = 0
    for j, t in enumerate(types):
        if strip_auto and t.startswith(AUTO_PREFIX):
            t = t[len(AUTO_PREFIX):]
            if t in target:
                auto_matched += 1
        if t and t in target and in_ds[j] > 0 and in_region[j]:
            factor[j] = min(max(target[t] / in_ds[j], factor_min), factor_max)
            scaled[j] = True
    W = sp.csr_matrix(sp.diags(factor.astype(np.float32)) @ g.W_chem, dtype=np.float32)
    W.eliminate_zeros()
    params = dict(g.params)
    q = np.quantile(factor[scaled], [0.05, 0.25, 0.5, 0.75, 0.95]) if scaled.any() else np.ones(5)
    params["normalize"] = {"ref": g_ref.dataset, "rule": NORM_RULE, "factor_min": factor_min, "factor_max": factor_max,
                           "median_over": MEDIAN_OVER, "strip_auto": bool(strip_auto),
                           "regions": sorted(regions) if regions is not None else None,
                           "auto_matched": auto_matched, "cells_outside_regions": int((~in_region).sum()),
                           "cells_scaled": int(scaled.sum()), "cells_unscaled": int((~scaled).sum()),
                           "factor_median": float(np.median(factor)),
                           "factor_scaled_quantiles_5_25_50_75_95": [round(float(x), 4) for x in q],
                           "at_min": int((scaled & (factor <= factor_min)).sum()),
                           "at_max": int((scaled & (factor >= factor_max)).sum()),
                           "types_in_ref": len(target)}
    return Graph(g.dataset, list(g.names), list(g.cell_class), list(g.transmitter), W, g.W_gap.copy(),
                 version="%s norm:%s" % (g.version, g_ref.dataset), files=dict(g.files), params=params)


def _cache_fresh(f):
    # the cache exists and (there is no database OR the cache is not older than the database) — use the
    # cache, don't rebuild an empty graph if the working database happens to be missing.
    return f.exists() and (not paths.DB_PATH.exists() or os.path.getmtime(f) >= os.path.getmtime(paths.DB_PATH))


def get(dataset_id, conn=None, normalize=None, norm_opts=None):
    """A dataset's graph from cache or from the database. normalize=<ref> — a graph with inputs
    normalized against a reference dataset ref (normalize_inputs), norm_opts = {strip_auto: bool,
    regions: [..]} (the YAML graph: key minus normalize), cached at data/cache/<ds>__norm-<ref>-<hash>.npz."""
    if normalize:
        opts = _norm_options(norm_opts)
        if _cache_fresh(_cache_file(dataset_id, normalize, opts)):
            g = load_cache(dataset_id, normalize, opts)
            if g is not None:
                return g
        g, g_ref = get(dataset_id, conn), get(normalize, conn)
        cell_regions = _cell_regions(conn, dataset_id, g.names) if opts["regions"] is not None else None
        gn = normalize_inputs(g, g_ref, _cell_types(conn, dataset_id, g.names), _cell_types(conn, normalize, g_ref.names),
                              strip_auto=opts["strip_auto"], regions=opts["regions"], cell_regions=cell_regions)
        save_cache(gn, normalize, opts)
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
