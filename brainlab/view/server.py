"""Local viewer: an http server on stdlib. Reads the database and results, computes nothing itself
(except the graph layout, which is cached for small datasets; a big dataset's subgraph is not
cached — it depends on the request)."""
import json
import mimetypes
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import networkx as nx
import numpy as np

from .. import paths
from ..lab import populations
from ..sim import result
from ..store import db, graph as graph_mod

STATIC = Path(__file__).parent / "static"

BIG_DATASET = 5000  # above this many cells we neither hand out nor lay out the whole graph


class BadRequest(Exception):
    """The request is syntactically valid but cannot be executed (needs focus/names) — 400, not 404."""


def layout(g):
    f = paths.CACHE / ("%s_layout.json" % g.dataset)
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    G = nx.Graph()
    G.add_nodes_from(range(g.n))
    c = (abs(g.W_chem) + g.W_gap).tocoo()
    G.add_weighted_edges_from((int(a), int(b), float(w)) for a, b, w in zip(c.row, c.col, c.data) if a != b)
    pos = nx.spring_layout(G, seed=0, k=1.5 / max(1, g.n) ** 0.5, iterations=100)
    out = {g.names[i]: [float(pos[i][0]), float(pos[i][1])] for i in range(g.n)}
    paths.CACHE.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out), encoding="utf-8")
    return out


def _datasets():
    conn = db.connect()
    try:
        rows = []
        for r in conn.execute("SELECT id FROM datasets"):
            ds = r["id"]
            n = conn.execute("SELECT COUNT(*) FROM neurons WHERE dataset=?", (ds,)).fetchone()[0]
            e = conn.execute("SELECT COUNT(*) FROM edges WHERE dataset=?", (ds,)).fetchone()[0]
            row = {"id": ds, "neurons": n, "edges": e}
            if n > BIG_DATASET:
                row["big"] = True
            rows.append(row)
        rows.sort(key=lambda r: r["neurons"])  # small datasets first, the page loads ds[0]
        return rows
    finally:
        conn.close()


def _dataset_exists(ds):
    conn = db.connect()
    try:
        return db.dataset_info(conn, ds) is not None
    finally:
        conn.close()


def _populations_payload(ds):
    if not _dataset_exists(ds):
        raise KeyError("no such dataset %s" % ds)
    groups = populations.resolve(ds)  # KeyError if lab/populations/<ds>.yaml is missing
    return {"groups": {name: len(names) for name, names in groups.items()},
            "state": populations.state_groups(ds)}


def _cell_types(ds, names):
    """cell_type for the given names — read straight from the database: the Graph/npz cache does
    not store it, and pulling the whole dataset through populations.resolve()/db.neurons() just for a subgraph is unnecessary."""
    conn = db.connect()
    try:
        out = {}
        names = list(names)
        for i in range(0, len(names), 400):
            chunk = names[i:i + 400]
            qs = ",".join("?" * len(chunk))
            rows = conn.execute(
                "SELECT name, cell_type FROM neurons WHERE dataset=? AND name IN (%s)" % qs, (ds, *chunk))
            for r in rows:
                out[r["name"]] = r["cell_type"]
        return out
    finally:
        conn.close()


def _subgraph_payload(g, ds, focus, top, min_count):
    fnames = populations.names(ds, focus)  # KeyError if the group does not exist
    if not fnames:
        raise KeyError("group %s in dataset %s is empty" % (focus, ds))
    focus_idx = g.idx(fnames)
    focus_set = set(int(i) for i in focus_idx)
    W = (abs(g.W_chem) + g.W_gap).tocsr()
    strength = np.asarray(W[focus_idx, :].sum(axis=0)).ravel() + np.asarray(W[:, focus_idx].sum(axis=1)).ravel()
    strength[focus_idx] = 0  # do not count the group's own cells as their own neighbors
    order = np.argsort(-strength)
    neighbors = [int(i) for i in order[:top] if strength[i] > 0]
    sel = sorted(focus_set | set(neighbors))
    sel_arr = np.array(sel, dtype=np.int64)
    sel_names = [g.names[i] for i in sel_arr]
    ctypes = _cell_types(ds, sel_names)

    Wsub = W[sel_arr][:, sel_arr].tocoo()
    G = nx.Graph()
    G.add_nodes_from(range(len(sel_arr)))
    G.add_weighted_edges_from((int(a), int(b), float(w)) for a, b, w in zip(Wsub.row, Wsub.col, Wsub.data) if a != b)
    pos = nx.spring_layout(G, seed=0, k=1.5 / max(1, len(sel_arr)) ** 0.5, iterations=100)

    nodes = []
    for li, gi in enumerate(sel_arr):
        gi = int(gi)
        p = pos.get(li, (0.0, 0.0))
        nm = g.names[gi]
        nodes.append({"name": nm, "class": g.cell_class[gi], "transmitter": g.transmitter[gi],
                      "type": ctypes.get(nm, ""), "focus": gi in focus_set,
                      "x": float(p[0]), "y": float(p[1])})

    edges = []
    c = g.W_chem[sel_arr][:, sel_arr].tocoo()
    for a, b, w in zip(c.row, c.col, c.data):
        if abs(w) >= min_count:
            edges.append({"pre": sel_names[b], "post": sel_names[a], "kind": "chemical", "count": abs(float(w)), "sign": 1 if w > 0 else -1})
    gp = g.W_gap[sel_arr][:, sel_arr].tocoo()
    for a, b, w in zip(gp.row, gp.col, gp.data):
        if a < b and w >= min_count:
            edges.append({"pre": sel_names[b], "post": sel_names[a], "kind": "electrical", "count": float(w), "sign": 0})
    return {"nodes": nodes, "edges": edges, "focus": focus, "group_size": len(fnames)}


def _graph_payload(ds, min_count, focus=None, top=300):
    if not _dataset_exists(ds):
        raise KeyError("no such dataset %s" % ds)
    g = graph_mod.get(ds)
    if g.n > BIG_DATASET:
        if not focus:
            raise BadRequest("dataset is big: specify focus=<group>")
        return _subgraph_payload(g, ds, focus, top, min_count)
    pos = layout(g)
    nodes = [{"name": n, "class": g.cell_class[i], "transmitter": g.transmitter[i], "x": pos[n][0], "y": pos[n][1]}
             for i, n in enumerate(g.names)]
    edges = []
    c = g.W_chem.tocoo()
    for a, b, w in zip(c.row, c.col, c.data):
        if abs(w) >= min_count:
            edges.append({"pre": g.names[b], "post": g.names[a], "kind": "chemical", "count": abs(float(w)), "sign": 1 if w > 0 else -1})
    gp = g.W_gap.tocoo()
    for a, b, w in zip(gp.row, gp.col, gp.data):
        if a < b and w >= min_count:
            edges.append({"pre": g.names[b], "post": g.names[a], "kind": "electrical", "count": float(w), "sign": 0})
    return {"nodes": nodes, "edges": edges}


def _run_payload(run_id, names=None):
    if "/" in run_id or "\\" in run_id or ".." in run_id:
        raise KeyError("invalid run_id: %r" % run_id)
    r = result.load(paths.RESULTS / run_id)
    if names:
        pos = {n: i for i, n in enumerate(r.names)}
        sel_names = [n for n in names if n in pos]
        idx = [pos[n] for n in sel_names]
        rates = r.rates[:, idx].tolist() if idx else [[] for _ in range(r.rates.shape[0])]
        return {"id": r.id, "names": sel_names, "window_ms": r.window_ms, "rates": rates,
                "flags": r.flags, "valence": r.valence, "extra": r.extra, "stimulus": r.stimulus}
    if len(r.names) > BIG_DATASET:
        raise BadRequest("dataset has more than %d cells: specify names=<a,b,c>" % BIG_DATASET)
    return {"id": r.id, "names": r.names, "window_ms": r.window_ms, "rates": r.rates.tolist(),
            "flags": r.flags, "valence": r.valence, "extra": r.extra, "stimulus": r.stimulus}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(STATIC), **kw)

    def log_message(self, *a):
        pass

    def _json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/api/datasets":
                return self._json(_datasets())
            if u.path.startswith("/api/populations/"):
                return self._json(_populations_payload(u.path.split("/")[3]))
            if u.path.startswith("/api/graph/"):
                focus = q.get("focus", [None])[0]
                try:
                    top = int(q.get("top", ["300"])[0])
                    min_count = float(q.get("min_count", ["2"])[0])
                except ValueError:
                    raise BadRequest("top/min_count must be numbers")
                return self._json(_graph_payload(u.path.split("/")[3], min_count, focus, top))
            if u.path == "/api/runs":
                return self._json(result.list_runs())
            if u.path.startswith("/api/run/"):
                names_param = q.get("names", [None])[0]
                names = [n for n in names_param.split(",") if n] if names_param else None
                return self._json(_run_payload(u.path.split("/")[3], names))
        except BadRequest as exc:
            return self._json({"error": str(exc)}, 400)
        except (KeyError, FileNotFoundError, ValueError) as exc:
            return self._json({"error": str(exc)}, 404)
        if u.path == "/":
            self.path = "/index.html"
        return super().do_GET()


def serve(port=8765, block=True):
    mimetypes.add_type("application/javascript", ".js")
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    if block:
        print("viewer: http://127.0.0.1:%d" % srv.server_address[1])
        srv.serve_forever()
        return srv
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


if __name__ == "__main__":
    serve()
