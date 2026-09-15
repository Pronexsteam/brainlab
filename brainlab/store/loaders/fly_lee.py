"""Three full adult fly scans in the Lee lab's shared format (open bucket, see fetch.FILES):
fafb_783 (FlyWire v783, female, brain), banc_888 (female, brain + ventral nerve cord), malecns_09 (male, CNS).

Assumptions:
- edge sign by the presynaptic cell's predicted transmitter (design §4): acetylcholine +1,
  GABA −1, glutamate −1 (GluCl in the fly), histamine −1 (in the fly, photoreceptors → lamina
  via histamine-chloride HisCl channels, inhibitory), monoamines (dopamine, serotonin, octopamine)
  and tyramine — MONOAMINE_SIGN (see below, cross-checked against the Shiu reference only for
  dopamine/serotonin/octopamine; tyramine is assigned to the same class by assumption
  and was not separately checked against the reference), "unclear"/empty/anything else not on the
  list — 0 (the edge stays in the database with a zero sign, does not enter W_chem);
- storage threshold min_count = 5 synapses per pair (like "connections" in Codex); the Shiu
  reference dataset stores everything, so edge counts between them cannot be compared directly;
- cell_class in the database = the meta super_class (top-level class); detailed cell_class/cell_sub_class
  are in extra;
- skipped_edges_unknown_id counts only edges that passed the min_count filter (whose pre or post
  was not found in meta) — this is a lower-bound estimate of the fraction of "lost" connectivity:
  edges below min_count are not counted here, even if their pre/post is also missing from meta.

Sign of monoamines (step 3b, cross-checked against the Shiu 2024 reference, flywire_630_shiu, id v630 = v783
for surviving cells): in FAFB v783 meta, cells with neurotransmitter_predicted in {dopamine, serotonin,
octopamine} number 8853; of those, ids were found as Presynaptic_ID in the reference parquet (2023_03_23_connectivity
_630_final.parquet) for 7373 cells, giving 656999 outgoing edges. Sign by the Excitatory column:
dopamine 570757 pos / 5 neg, serotonin 78202 pos / 125 neg, octopamine 7859 pos / 51 neg;
combined 656818 pos / 181 neg, positive fraction 0.9997 (per transmitter separately —
0.9999 / 0.9984 / 0.9936). In the reference, monoamines are almost unambiguously positive (the rare
negatives are prediction noise on individual cells), so MONOAMINE_SIGN = 1. This is a cross-check of
the assumption against third-party data, not a fit to the gates (monoamines are not part of gates 1/2).

The edge list is read lazily in batches via pyarrow (3.2 GB for MaleCNS): the open file/memory_map
is closed as soon as the edge generator is exhausted or interrupted by an exception.
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
MONOAMINE_SIGN = 1            # cross-checked in task 4, step 3b against the Shiu reference (see the docstring above)
SIGN = {"acetylcholine": 1, "gaba": -1, "glutamate": -1, "histamine": -1}
MONOAMINES = ("dopamine", "serotonin", "octopamine", "tyramine")   # -> MONOAMINE_SIGN (see the assumptions)
SIGN_RULE = ("presynaptic neurotransmitter_predicted: ach +1, gaba -1, glutamate -1, histamine -1, "
             "dopamine/serotonin/octopamine/tyramine MONOAMINE_SIGN, unclear/unknown/empty 0")
EXTRA_COLS = ["cell_class", "cell_sub_class", "cell_function", "cell_function_detailed", "flow",
              "neurotransmitter_score", "fafb_cell_type", "body_part_sensory", "body_part_effector"]
_NA_STRINGS = ("", "nan", "none", "<na>")   # case-insensitive: missing-value sentinels in Arrow string columns


def _s(v):
    if v is None:
        return ""
    if isinstance(v, float) and np.isnan(v):
        return ""
    s = str(v)
    return "" if s.strip().lower() in _NA_STRINGS else s


def rule_hash(min_count):
    """Short hash of the sign rule + threshold: changes together with SIGN_RULE/min_count/
    MONOAMINE_SIGN, so a rule change changes the dataset's version in the database (task 1 of the final wave)."""
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
    """Lazy generator of edge batches: memory_map/reader are opened once and closed
    (finally) as soon as the source is exhausted or interrupted by an exception. We try the IPC file
    format first (it is what all three Lee-lab datasets use as of 2026-09-14); on ArrowInvalid we fall
    back to IPC stream (batches there are read one at a time without random access, also lazily)."""
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


def load(conn, raw_dir, dataset_id, min_count=5, store_as=None):
    """Load one Lee-lab scan into the database. dataset_id selects the files (SPECS); store_as, if
    given, is the id the dataset is registered under (e.g. fafb_783_all for the same files at
    min_count=1), with params.min_count recording the threshold actually applied."""
    spec = SPECS[dataset_id]
    dataset_id = store_as or dataset_id
    raw_dir = Path(raw_dir)
    meta_f, edge_f = raw_dir / spec["meta"], raw_dir / spec["edges"]
    meta = ft.read_table(meta_f).to_pandas()
    idc = spec["id_col"]
    if idc not in meta.columns:
        raise ValueError("%s has no column %s" % (meta_f.name, idc))
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
        raise ValueError("%s is missing columns %s (has: %s)" % (edge_f.name, sorted(need - set(schema.names)), schema.names))
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
        # one transaction per loader (see worm_cook2019.load): an edge generator that fails halfway
        # through (including inside _edge_batches) rolls back both register_dataset and add_neurons — the dataset is absent from the database
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
    import argparse
    from .. import fetch
    ap = argparse.ArgumentParser(description="Load Lee-lab fly scans into the database.")
    ap.add_argument("datasets", nargs="*", help="dataset ids (default: all of %s)" % ", ".join(SPECS))
    ap.add_argument("--min-count", type=int, default=5, help="synapse threshold per pair (default 5)")
    ap.add_argument("--as", dest="store_as", default=None,
                    help="register under this id instead of the dataset id (one dataset only)")
    a = ap.parse_args()
    if a.store_as and len(a.datasets) != 1:
        ap.error("--as needs exactly one dataset")
    c = db.connect()
    for ds in a.datasets or list(SPECS):
        print(a.store_as or ds, load(c, fetch.raw_dir(ds), ds, min_count=a.min_count, store_as=a.store_as))
    c.close()
