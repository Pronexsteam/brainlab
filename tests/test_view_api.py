import json
import urllib.error
import urllib.request

import pytest

from brainlab import paths
from brainlab.view import server


def _get(port, path):
    with urllib.request.urlopen("http://127.0.0.1:%d%s" % (port, path), timeout=10) as r:
        return r.status, r.read()


@pytest.fixture(scope="module")
def srv():
    if not paths.DB_PATH.exists():
        pytest.skip("no working database data/store.sqlite")
    s = server.serve(port=0, block=False)
    yield s
    s.shutdown()
    s.server_close()


def test_datasets_and_graph(srv):
    port = srv.server_address[1]
    st, body = _get(port, "/api/datasets")
    ds = json.loads(body)
    assert st == 200 and any(d["id"] == "worm_cook2019" for d in ds)
    st, body = _get(port, "/api/graph/worm_cook2019?min_count=3")
    g = json.loads(body)
    assert st == 200 and len(g["nodes"]) > 300 and len(g["edges"]) > 500
    node = next(n for n in g["nodes"] if n["name"] == "AVAL")
    assert node["class"] == "neuron" and isinstance(node["x"], float)


def test_runs_and_index(srv):
    port = srv.server_address[1]
    st, body = _get(port, "/api/runs")
    assert st == 200 and isinstance(json.loads(body), list)
    st, body = _get(port, "/")
    assert st == 200 and b"<canvas" in body


def test_datasets_ordered_small_first(srv):
    port = srv.server_address[1]
    st, body = _get(port, "/api/datasets")
    ds = json.loads(body)
    counts = [d["neurons"] for d in ds]
    assert counts == sorted(counts)
    assert ds[0]["id"] == "worm_cook2019"  # the smallest dataset — the page loads it first
    assert not ds[0].get("big")
    big = [d for d in ds if d["id"] != "worm_cook2019"]
    if big:
        assert all(d.get("big") for d in big)  # fly datasets (>5000 cells) are flagged


def test_unknown_dataset_is_404(srv):
    port = srv.server_address[1]
    try:
        _get(port, "/api/graph/nope")
        assert False, "should have raised 404"
    except urllib.error.HTTPError as e:
        assert e.code == 404
        body = json.loads(e.read())
        assert "error" in body
