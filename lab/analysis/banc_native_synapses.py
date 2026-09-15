"""Check: is the BANC v888 synapse shortfall (0.39 of FAFB for cells on the sugar → MN9 path) in the
native synapse table, or only in Lee-lab's compiled edgelist_simple_v3?

Run from the project root:
    python lab/analysis/banc_native_synapses.py                # scan both native tables + report (~2-3 min)
    python lab/analysis/banc_native_synapses.py --stage report # report only, from the scan cache (--cache)

Reads: data/banc_888/raw/banc_888_synapses_v3_enriched.parquet (199 million rows, by row group, 4 columns),
data/fafb_783/raw/fafb_783_synapses.parquet (55.6 million), both Lee-lab edgelists, meta files and
data/store.sqlite (cell labels). Changes nothing under brainlab/ or data/; intermediate histograms go
into --cache (a temp folder by default). Report: docs/2026-09-15-banc-native-synapse-check.md.
"""
import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as ft
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from brainlab.store import db  # noqa: E402

RAW = {"banc_888": ROOT / "data/banc_888/raw", "fafb_783": ROOT / "data/fafb_783/raw"}
SYN = {"banc_888": "banc_888_synapses_v3_enriched.parquet", "fafb_783": "fafb_783_synapses.parquet"}
EDGES = {"banc_888": "banc_888_edgelist_simple_v3.feather", "fafb_783": "fafb_783_simple_edgelist.feather"}
META = {"banc_888": "banc_888_meta.feather", "fafb_783": "fafb_783_meta.feather"}
IDCOL = {"banc_888": "banc_888_id", "fafb_783": "fafb_783_id"}
# native columns: (pre, post, score); BANC also has size
COLS = {"banc_888": ("pre_root_id", "post_root_id", "mean_score"), "fafb_783": ("pre", "post", "confidence")}
PATH_TYPES = ["MN9", "DNge062", "DNge080", "DNge059", "CB0553", "CB0493", "CB0824", "CB0393", "CB0616"]
# histograms: BANC mean_score — step 0.001 over [0, 1); size — step 1 over [0, 1000); FAFB confidence — integers 0..255
SB_N, SB_W = 1000, 0.001
SZ_N = 1000
CF_N = 256
# per-cell histograms (more compact): BANC score step 0.005 over [0, 0.32) + tail; FAFB confidence 0..255
CELL_B, CELL_W = 65, 0.005

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- cell labels
def cells(ds):
    conn = db.connect()
    ns = db.neurons(conn, ds)
    conn.close()
    rows = []
    for x in ns:
        ft_ = x["extra"].get("fafb_cell_type", "")
        rows.append({"id": int(x["name"]), "cell_type": x["cell_type"], "fafb_cell_type": ft_,
                     "side": x["side"], "region": x["region"], "cell_class": x["cell_class"]})
    df = pd.DataFrame(rows).set_index("id")
    # path type: in BANC by cell_type or fafb_cell_type (exact match, no auto:), in FAFB by cell_type
    t = df["cell_type"].where(df["cell_type"].isin(PATH_TYPES), df["fafb_cell_type"])
    df["ptype"] = t.where(t.isin(PATH_TYPES), "")
    return df


# ---------------------------------------------------------------- native table scan
def scan(ds, meta, cache):
    """One pass over the parquet: global score histograms (all / post∈meta / pre,post∈meta), a size
    histogram (BANC), per-cell score histograms for post∈meta (pre∈meta and pre∉meta separately), and full
    rows for post ∈ path cells."""
    out = cache / ("%s_scan.npz" % ds)
    if out.exists():
        log(ds, "scan from cache", out)
        return dict(np.load(out, allow_pickle=True))
    pre_c, post_c, sc_c = COLS[ds]
    cols = list(COLS[ds]) + (["size"] if ds == "banc_888" else [])
    ids = pd.Index(meta.index.values)                      # int64
    n_cells = len(ids)
    path_ids = set(meta.index[meta["ptype"] != ""].tolist())
    path_arr = np.fromiter(path_ids, dtype=np.int64)
    is_banc = ds == "banc_888"
    NB = SB_N if is_banc else CF_N
    g_all = np.zeros(NB, np.int64)
    g_post = np.zeros(NB, np.int64)
    g_both = np.zeros(NB, np.int64)
    g_size_both = np.zeros(SZ_N, np.int64)
    g_size_all = np.zeros(SZ_N, np.int64)
    CB = CELL_B if is_banc else CF_N
    cell_in = np.zeros(n_cells * CB, np.int64)             # post∈meta, pre∈meta
    cell_out = np.zeros(n_cells * CB, np.int64)            # post∈meta, pre∉meta
    keep = []
    f = pq.ParquetFile(RAW[ds] / SYN[ds])
    n_rg = f.metadata.num_row_groups
    total = 0
    t0 = time.time()
    for i in range(n_rg):
        t = f.read_row_group(i, columns=cols)
        pre = pc.cast(t[pre_c], pa.int64()).to_numpy()
        post = pc.cast(t[post_c], pa.int64()).to_numpy()
        sc = t[sc_c].to_numpy()
        total += len(sc)
        if is_banc:
            b = np.minimum((sc / SB_W).astype(np.int64), SB_N - 1)
            cb = np.minimum((sc / CELL_W).astype(np.int64), CB - 1)
            sz = np.minimum(t["size"].to_numpy().astype(np.int64), SZ_N - 1)
            g_size_all += np.bincount(sz, minlength=SZ_N)
        else:
            b = np.clip(sc.astype(np.int64), 0, NB - 1)
            cb = b
        g_all += np.bincount(b, minlength=NB)
        pi = ids.get_indexer(post)
        m_post = pi >= 0
        m_pre = ids.get_indexer(pre) >= 0
        g_post += np.bincount(b[m_post], minlength=NB)
        both = m_post & m_pre
        g_both += np.bincount(b[both], minlength=NB)
        if is_banc:
            g_size_both += np.bincount(sz[both], minlength=SZ_N)
        cell_in += np.bincount(pi[both] * CB + cb[both], minlength=n_cells * CB)
        m_o = m_post & ~m_pre
        cell_out += np.bincount(pi[m_o] * CB + cb[m_o], minlength=n_cells * CB)
        mp = np.isin(post, path_arr)
        if mp.any():
            d = {"pre": pre[mp], "post": post[mp], "score": sc[mp], "pre_in_meta": m_pre[mp]}
            if is_banc:
                d["size"] = t["size"].to_numpy()[mp]
            keep.append(pd.DataFrame(d))
        if i % 200 == 0:
            log(ds, "row group %d/%d, rows %d, %.0f s" % (i, n_rg, total, time.time() - t0))
    keep = pd.concat(keep, ignore_index=True)
    res = {"total": total, "g_all": g_all, "g_post": g_post, "g_both": g_both, "g_size_all": g_size_all,
           "g_size_both": g_size_both, "cell_in": cell_in.reshape(n_cells, CB), "cell_out": cell_out.reshape(n_cells, CB),
           "cell_ids": ids.values, "path_rows": keep.to_records(index=False)}
    np.savez(out, **res)
    log(ds, "scan done: %d rows in %.0f s; rows for path cells %d" % (total, time.time() - t0, len(keep)))
    return res


# ---------------------------------------------------------------- Lee-lab edgelist
def edgelist(ds, meta):
    t = ft.read_table(RAW[ds] / EDGES[ds])
    pre = pc.cast(t["pre"], pa.int64()).to_numpy()
    post = pc.cast(t["post"], pa.int64()).to_numpy()
    cnt = t["count"].to_numpy().astype(np.int64)
    df = pd.DataFrame({"pre": pre, "post": post, "count": cnt})
    tot_col = "post_count" if "post_count" in t.column_names else "total_input"
    df["total_input"] = t[tot_col].to_numpy().astype(np.int64)
    ids = pd.Index(meta.index.values)
    info = {"rows": len(df), "sum_count": int(cnt.sum()), "pre_in_meta": float((ids.get_indexer(pre) >= 0).mean()),
            "post_in_meta": float((ids.get_indexer(post) >= 0).mean()), "min_count": int(cnt.min()),
            "n_count1": int((cnt == 1).sum())}
    per_post = df.groupby("post")["count"].sum()
    ti = df.groupby("post")["total_input"].first()
    info["total_input_eq_sum"] = float((per_post.reindex(ti.index).values == ti.values).mean())
    info["dup_pairs"] = int(df.duplicated(["pre", "post"]).sum())
    return df, per_post, info


# ---------------------------------------------------------------- report
def cum_ge(hist):
    """hist by increasing threshold → number of rows with bin ≥ k."""
    return np.cumsum(hist[::-1])[::-1]


def fmt_thr(ds, k):
    return ("%.3f" % (k * SB_W)) if ds == "banc_888" else str(k)


def find_thr(ds, hist, target):
    c = cum_ge(hist)
    k = int(np.argmin(np.abs(c - target)))
    return k, int(c[k])


def per_cell_counts(res, ds, thr_bin_cell):
    """Inputs per cell (post∈meta): pre∈meta and total, at a threshold from the per-cell histogram."""
    ci, co = res["cell_in"], res["cell_out"]
    a = ci[:, thr_bin_cell:].sum(1)
    b = co[:, thr_bin_cell:].sum(1)
    return pd.DataFrame({"in_meta": a, "all": a + b}, index=res["cell_ids"])


def cell_bin(ds, thr):
    return int(round(thr / CELL_W)) if ds == "banc_888" else int(thr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "scan", "report"])
    ap.add_argument("--cache", default=str(Path(tempfile.gettempdir()) / "brainlab_banc_native"))
    a = ap.parse_args()
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    meta = {ds: cells(ds) for ds in RAW}
    res = {ds: scan(ds, meta[ds], cache) for ds in RAW}
    if a.stage == "scan":
        return
    el, per_post, einfo = {}, {}, {}
    for ds in RAW:
        el[ds], per_post[ds], einfo[ds] = edgelist(ds, meta[ds])

    print("\n" + "=" * 100 + "\n### 1. Whole file: native table vs edgelist\n")
    thr = {}
    for ds in RAW:
        r, e = res[ds], einfo[ds]
        n_all, n_post, n_both = int(r["g_all"].sum()), int(r["g_post"].sum()), int(r["g_both"].sum())
        print("%s: native rows %d; post∈meta %d (%.1f %%); pre,post∈meta %d (%.1f %%); edgelist Σcount %d (%.1f %% of native, %.1f %% of pre,post∈meta)"
              % (ds, n_all, n_post, 100 * n_post / n_all, n_both, 100 * n_both / n_all, e["sum_count"],
                 100 * e["sum_count"] / n_all, 100 * e["sum_count"] / n_both))
        print("   edgelist: rows %d, min count %d, pairs with count=1: %d, pre∈meta %.4f, post∈meta %.4f, total_input==Σcount by post: %.4f"
              % (e["rows"], e["min_count"], e["n_count1"], e["pre_in_meta"], e["post_in_meta"], e["total_input_eq_sum"]))
        h = r["g_all"]
        nz = np.nonzero(h)[0]
        print("   score: min %s, max %s; duplicate pairs in edgelist %d" % (fmt_thr(ds, nz[0]), fmt_thr(ds, nz[-1]), e["dup_pairs"]))
        lo = nz[0]
        print("   lower edge of the score distribution (all rows), first 12 bins: " +
              ", ".join("%s:%d" % (fmt_thr(ds, k), h[k]) for k in range(lo, lo + 12)))
        for name, hh in (("all", r["g_all"]), ("post∈meta", r["g_post"]), ("pre,post∈meta", r["g_both"])):
            k, c = find_thr(ds, hh, e["sum_count"])
            print("   score threshold giving Σ edgelist among \"%s\": score ≥ %s → %d (Δ %+d)" % (name, fmt_thr(ds, k), c, c - e["sum_count"]))
        thr[ds] = find_thr(ds, r["g_both"], e["sum_count"])[0]
        # score quantiles
        c = np.cumsum(r["g_both"]) / n_both
        qs = [np.searchsorted(c, q) for q in (0.05, 0.25, 0.5, 0.75, 0.95)]
        print("   score quantiles (pre,post∈meta) 5/25/50/75/95: " + " / ".join(fmt_thr(ds, k) for k in qs))
        if ds == "banc_888":
            sz = r["g_size_both"]
            c = np.cumsum(sz) / sz.sum()
            print("   size (pre,post∈meta): min %d, quantiles 5/25/50/75/95: %s" % (np.nonzero(sz)[0][0], " / ".join(str(np.searchsorted(c, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95))))
            k, cc = find_thr(ds, sz, e["sum_count"])
            print("   size threshold giving Σ edgelist among pre,post∈meta: size ≥ %d → %d" % (k, cc))

    print("\n### 1b. Per-cell cross-check: a cell's edgelist input vs native (post∈meta) at different thresholds\n")
    best = {}
    for ds in RAW:
        r = res[ds]
        pp = per_post[ds]
        cands = [0.0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20] if ds == "banc_888" else [0, 50, 60, 80, 100, 120]
        rows = []
        for thr_ in cands:
            pc_ = per_cell_counts(r, ds, cell_bin(ds, thr_))
            e = pp.reindex(pc_.index).fillna(0).astype(int)
            for col in ("in_meta", "all"):
                nat = pc_[col]
                m = (e > 0) | (nat > 0)
                exact = float((nat[m] == e[m]).mean())
                ratio = (e[m] + 0.5) / (nat[m] + 0.5)
                rows.append({"threshold": thr_, "pre": col, "cells": int(m.sum()), "exact_match": round(exact, 4),
                             "median edgelist/native": round(float(np.median(ratio)), 3),
                             "Σ native": int(nat[m].sum()), "Σ edgelist": int(e[m].sum())})
        df = pd.DataFrame(rows)
        print(ds); print(df.to_string(index=False)); print()
        b = df[df["pre"] == "in_meta"].sort_values("exact_match", ascending=False).iloc[0]
        best[ds] = float(b["threshold"])
        print("   best threshold (pre∈meta): %s, exact match %.1f %% of cells" % (b["threshold"], 100 * b["exact_match"]))

    print("\n### 1c. Median input per cell (central_brain, cells with ≥1 native input): native-all / native at threshold / edgelist\n")
    cb_med = {}
    for ds in RAW:
        r, pp = res[ds], per_post[ds]
        cbm = meta[ds]["region"] == "central_brain"
        nat0 = per_cell_counts(r, ds, 0)
        natT = per_cell_counts(r, ds, cell_bin(ds, best[ds]))
        idx = meta[ds].index[cbm]
        e = pp.reindex(idx).fillna(0)
        n0, nT = nat0.loc[idx, "all"], natT.loc[idx, "in_meta"]
        m = n0 > 0
        cb_med[ds] = (int(np.median(n0[m])), int(np.median(nT[m])), int(np.median(e[m])), int(m.sum()))
        print("%s: cells %d; median native-all %d, native(pre∈meta, score≥%s) %d, edgelist %d" % ((ds, cb_med[ds][3], cb_med[ds][0], best[ds]) + cb_med[ds][1:3]))

    print("\n### 2. Path cells: native inputs vs edgelist\n")
    tabs = {}
    for ds in RAW:
        r = res[ds]
        pr = pd.DataFrame(r["path_rows"])
        pp = per_post[ds]
        m = meta[ds]
        rows = []
        for cid, g in pr.groupby("post"):
            e = int(pp.get(cid, 0))
            d = {"id": cid, "type": m.at[cid, "ptype"], "side": (m.at[cid, "side"] or "?")[:1], "native_all": len(g),
                 "native_pre∈meta": int(g["pre_in_meta"].sum()),
                 "native≥thr": int((g["pre_in_meta"] & (g["score"] >= best[ds])).sum()), "edgelist": e}
            if ds == "banc_888":
                d["score_q50"] = round(float(g["score"].median()), 3)
                d["size_q50"] = int(g["size"].median())
            else:
                d["conf_q50"] = int(g["score"].median())
            rows.append(d)
        df = pd.DataFrame(rows).sort_values(["type", "side"])
        if ds == "banc_888":
            mm = ft.read_table(RAW[ds] / META[ds], columns=[IDCOL[ds], "input_connections", "proofread", "status"]).to_pandas()
            mm.index = mm[IDCOL[ds]].astype(np.int64)
            df["meta_input"] = df["id"].map(mm["input_connections"]).fillna(-1).astype(int)
            df["proofread"] = df["id"].map(mm["proofread"])
            df["status"] = df["id"].map(mm["status"]).fillna("").str.slice(0, 28)
        tabs[ds] = df
        print(ds, "(edgelist threshold:", best[ds], ")"); print(df.to_string(index=False)); print()

    print("\n### 3. BANC / FAFB comparison by type (mean over cells of the type)\n")
    rows = []
    for t in PATH_TYPES:
        b, f = tabs["banc_888"], tabs["fafb_783"]
        b, f = b[b["type"] == t], f[f["type"] == t]
        if len(b) == 0 or len(f) == 0:
            rows.append({"type": t, "BANC cells": len(b), "FAFB cells": len(f)}); continue
        rows.append({"type": t, "BANC cells": len(b), "BANC native": int(b["native_all"].mean()),
                     "BANC edgelist": int(b["edgelist"].mean()), "FAFB cells": len(f), "FAFB native": int(f["native_all"].mean()),
                     "FAFB edgelist": int(f["edgelist"].mean()), "native B/F": round(b["native_all"].mean() / f["native_all"].mean(), 2),
                     "edgelist B/F": round(b["edgelist"].mean() / max(f["edgelist"].mean(), 1), 2),
                     "BANC edgelist/native": round(b["edgelist"].mean() / b["native_all"].mean(), 2),
                     "FAFB edgelist/native": round(f["edgelist"].mean() / f["native_all"].mean(), 2)})
    cmp_ = pd.DataFrame(rows)
    print(cmp_.to_string(index=False))
    ok = cmp_.dropna()
    print("\nmedian over types: native B/F %.2f, edgelist B/F %.2f" % (ok["native B/F"].median(), ok["edgelist B/F"].median()))

    print("\n### 3b. Matched filter: top X %% of synapses by score in each dataset (pre,post∈meta)\n")
    rows = []
    for frac in (1.0, 0.75, 0.5, 0.25):
        row = {"frac": frac}
        for ds in RAW:
            r = res[ds]
            c = np.cumsum(r["g_both"]) / r["g_both"].sum()
            k = int(np.searchsorted(c, 1 - frac))
            thr_ = k * SB_W if ds == "banc_888" else k
            pr = pd.DataFrame(r["path_rows"])
            pr = pr[pr["pre_in_meta"] & (pr["score"] >= thr_)]
            per = pr.groupby("post").size()
            m = meta[ds]
            typ = m.loc[per.index, "ptype"]
            row[ds + " threshold"] = fmt_thr(ds, k)
            for t in PATH_TYPES:
                v = per[typ.values == t]
                row["%s %s" % (ds[:4], t)] = int(v.mean()) if len(v) else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    rat = {t: [round(df["banc %s" % t][i] / df["fafb %s" % t][i], 2) if df["fafb %s" % t][i] > 0 else np.nan for i in range(len(df))] for t in PATH_TYPES}
    print("\nBANC/FAFB ratio by type at fractions", list(df["frac"]))
    for t, v in rat.items():
        print("  %-8s %s" % (t, v))

    print("\n### 4. pre→post pairs for MN9 and DNge062: edgelist count vs native (pre∈meta, score≥threshold)\n")
    for ds in RAW:
        r = res[ds]
        pr = pd.DataFrame(r["path_rows"])
        m = meta[ds]
        sel = pr[pr["post"].map(m["ptype"]).isin(["MN9", "DNge062"]) & pr["pre_in_meta"] & (pr["score"] >= best[ds])]
        nat = sel.groupby(["pre", "post"]).size().rename("native")
        e = el[ds].groupby(["pre", "post"])["count"].sum()     # the FAFB edgelist has duplicate pairs
        j = pd.concat([nat, e.reindex(nat.index)], axis=1).fillna(0).astype(int)
        j = j.rename(columns={"count": "edgelist"})
        j["pre_type"] = [m["ptype"].get(p, "") or m["cell_type"].get(p, "") for p, _ in j.index]
        print(ds, ": pairs %d, exact match %.1f %%, Σ native %d, Σ edgelist %d" % (len(j), 100 * (j["native"] == j["edgelist"]).mean(), j["native"].sum(), j["edgelist"].sum()))
        print(j.sort_values("native", ascending=False).head(12).to_string())
        # pairs present in edgelist but absent from the native table (for the same post cells)
        posts = set(sel["post"].unique())
        ee = e[[p_ in posts for _, p_ in e.index]]
        miss = ee[~ee.index.isin(nat.index)]
        print("   pairs in edgelist for these cells: %d (Σ %d); of those, missing from the native table: %d (Σ %d)" % (len(ee), ee.sum(), len(miss), miss.sum()))
        if len(miss):
            mi = pd.DataFrame({"count": miss.values}, index=miss.index)
            mi["pre_class"] = [meta[ds]["cell_class"].get(p_, "?") for p_, _ in mi.index]
            mi["pre_region"] = [meta[ds]["region"].get(p_, "?") for p_, _ in mi.index]
            print(mi.sort_values("count", ascending=False).head(8).to_string())
            print("   pre classes of the missing pairs:", mi.groupby("pre_class")["count"].sum().sort_values(ascending=False).head(6).to_dict())
        print()


if __name__ == "__main__":
    main()
