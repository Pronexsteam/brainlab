"""The Shiu et al. 2024 reference dataset: FlyWire v630 as in the philshiu/Drosophila_brain_model repository.

We take the files as-is: all synapses with no threshold, edge sign = the Excitatory column (by the
authors, from the presynaptic cell's predicted transmitter). Cells are unlabeled (id only); groups
for experiments are explicit id lists in lab/populations/flywire_630_shiu.yaml, taken from the
authors' figures.ipynb. The dataset has one purpose: our LIF must reproduce the reference on this
same data (gate 2).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .. import db

DATASET_ID = "flywire_630_shiu"
SOURCE = "https://github.com/philshiu/Drosophila_brain_model (Shiu et al. 2024, Nature; MIT)"
LICENSE = "MIT (repository code and tables); FlyWire CC-BY 4.0"
VERSION = "630-shiu-2023_03_23"
MIN_COUNT = 1
SIGN_RULE = "sign column Excitatory from Shiu 2024 parquet (per edge)"


def load(conn, raw_dir):
    raw_dir = Path(raw_dir)
    comp_f = raw_dir / "2023_03_23_completeness_630_final.csv"
    con_f = raw_dir / "2023_03_23_connectivity_630_final.parquet"
    comp = pd.read_csv(comp_f)
    ids = comp.iloc[:, 0].astype(np.int64).astype(str).tolist()
    con = pd.read_parquet(con_f, columns=["Presynaptic_ID", "Postsynaptic_ID", "Connectivity", "Excitatory"])
    pre = con["Presynaptic_ID"].astype(np.int64).astype(str).to_numpy()
    post = con["Postsynaptic_ID"].astype(np.int64).astype(str).to_numpy()
    cnt = con["Connectivity"].to_numpy(dtype=np.float32)
    sgn = con["Excitatory"].to_numpy(dtype=np.int64)
    pos = pd.Series(sgn > 0).groupby(pre).sum()
    neg = pd.Series(sgn < 0).groupby(pre).sum()
    files = {p.name: db.sha256_file(p) for p in (comp_f, con_f)}
    rows = []
    for name in ids:
        p, q = int(pos.get(name, 0)), int(neg.get(name, 0))
        tr = "" if p == q == 0 else ("exc" if q == 0 else ("inh" if p == 0 else "mixed"))
        rows.append({"name": name, "cell_type": "", "cell_class": "neuron", "transmitter": tr, "side": "",
                     "region": "", "extra": {"signs": {"pos": p, "neg": q}} if tr == "mixed" else {}})
    try:
        # one transaction per loader (see worm_cook2019.load): exception — rollback, dataset absent from the database
        db.register_dataset(conn, DATASET_ID, SOURCE, VERSION, LICENSE, files,
                            {"sign_rule": SIGN_RULE, "min_count": MIN_COUNT, "source_files": sorted(files)})
        db.add_neurons(conn, DATASET_ID, rows)
        db.add_edges(conn, DATASET_ID, ({"pre": a, "post": b, "kind": "chemical", "count": float(c), "sign": int(s)}
                                        for a, b, c, s in zip(pre, post, cnt, sgn)))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return {"neurons": len(ids), "edges": int(len(con)), "positive": int((sgn > 0).sum()), "negative": int((sgn < 0).sum())}


if __name__ == "__main__":
    from .. import fetch
    c = db.connect()
    print(load(c, fetch.raw_dir(DATASET_ID)))
    c.close()
