"""C. elegans worm, hermaphrodite, Cook et al. 2019 wiring diagram (file herm_full_edgelist.csv from OpenWorm/c302).

Assumptions:
- cell class: "neuron" if the name is present in the White 1986 diagram (aconnectome_white_1986_whole.csv)
  or in the list of 20 pharyngeal neurons; otherwise "end_organ" (muscles, glands, etc.);
- sign of a chemical connection, by the presynaptic cell: GABAergic (the classic list of 26 cells
  per Gendrel et al. 2016) → −1, everything else → +1. We do not account for inhibitory glutamate
  (GluCl) — this is the sign_rule parameter in the database, it can be changed;
- gap junctions (electrical) are unsigned, weight = number of contacts.

A naming mismatch between the files: herm_full_edgelist.csv writes motor-neuron numbers
(AS, DA, DB, DD, VA, VB, VC, VD) with a leading zero (DD01, VD01, ...), while
aconnectome_white_1986_whole.csv and the GABA cell list do not (DD1, VD1, ...).
Without normalization, 66 motor neurons (DD1-6, VD1-13, VA1-9, VB1-9, VC1-6, DA1-9,
DB1-7, AS1-9) ended up in "end_organ", and the GABA edge count dropped from 26 to 11.
The leading zero is stripped (_normalize) before classification and before writing to the database.
"""
import csv
import re
from pathlib import Path

from .. import db

DATASET_ID = "worm_cook2019"
SOURCE = "https://github.com/openworm/c302 (c302/data/herm_full_edgelist.csv, Cook et al. 2019)"
LICENSE = "CC-BY (OpenWorm)"

PHARYNGEAL = frozenset("I1L I1R I2L I2R I3 I4 I5 I6 M1 M2L M2R M3L M3R M4 M5 MCL MCR MI NSML NSMR".split())
GABA_NEURONS = frozenset(
    ["DD%d" % i for i in range(1, 7)] + ["VD%d" % i for i in range(1, 14)]
    + ["RMEL", "RMER", "RMED", "RMEV", "AVL", "DVB", "RIS"])
SIGN_RULE = "GABA presynaptic -> -1, else +1; electrical unsigned"

_ZERO_PAD = re.compile(r"^([A-Za-z]+)(0\d+)$")


def _normalize(name):
    """Strips the leading zero in a cell number: DD01 -> DD1 (see the assumptions above)."""
    m = _ZERO_PAD.match(name)
    return m.group(1) + str(int(m.group(2))) if m else name


def _white_neurons(raw_dir):
    names = set()
    path = Path(raw_dir) / "aconnectome_white_1986_whole.csv"
    if not path.exists():
        return names
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            names.add(row["pre"].strip())
            names.add(row["post"].strip())
    return names


def load(conn, raw_dir):
    raw_dir = Path(raw_dir)
    edge_file = raw_dir / "herm_full_edgelist.csv"
    neuron_names = _white_neurons(raw_dir) | PHARYNGEAL
    nodes, edges = {}, []
    with open(edge_file, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pre, post = _normalize(row["Source"].strip()), _normalize(row["Target"].strip())
            kind = row["Type"].strip().lower()
            if kind not in ("chemical", "electrical"):
                continue
            count = float(row["Weight"])
            for n in (pre, post):
                nodes.setdefault(n, "neuron" if n in neuron_names else "end_organ")
            sign = 0 if kind == "electrical" else (-1 if pre in GABA_NEURONS else 1)
            edges.append({"pre": pre, "post": post, "kind": kind, "count": count, "sign": sign})
    files = {p.name: db.sha256_file(p) for p in (edge_file, raw_dir / "aconnectome_white_1986_whole.csv") if p.exists()}
    rows = []
    for name, cls in sorted(nodes.items()):
        side = "L" if name.endswith("L") and cls == "neuron" else ("R" if name.endswith("R") and cls == "neuron" else "")
        rows.append({"name": name, "cell_type": name.rstrip("LR0123456789") if cls == "neuron" else "",
                     "cell_class": cls, "transmitter": "GABA" if name in GABA_NEURONS else "",
                     "side": side, "region": "pharynx" if name in PHARYNGEAL else "", "extra": {}})
    try:
        # register_dataset/add_neurons/add_edges do not commit themselves — one transaction per loader,
        # rolled back on exception: the dataset is either loaded in full or absent from the database (task 5 of the final wave)
        db.register_dataset(conn, DATASET_ID, SOURCE, "cook2019-c302", LICENSE, files,
                            {"sign_rule": SIGN_RULE, "gaba_count": len(GABA_NEURONS), "pharyngeal": sorted(PHARYNGEAL)})
        db.add_neurons(conn, DATASET_ID, rows)
        db.add_edges(conn, DATASET_ID, edges)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return {"neurons": sum(1 for c in nodes.values() if c == "neuron"),
            "other": sum(1 for c in nodes.values() if c == "end_organ"),
            "edges": len(edges), "gaba": sum(1 for n in nodes if n in GABA_NEURONS)}


if __name__ == "__main__":
    from .. import fetch
    c = db.connect()
    print(load(c, fetch.raw_dir("worm")))
    c.close()
