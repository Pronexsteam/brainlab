import json
import urllib.error
import urllib.parse
import urllib.request

import pytest

from brainlab import paths
from brainlab.view import server

playwright = pytest.importorskip("playwright.sync_api")


@pytest.fixture(scope="module")
def srv():
    if not paths.DB_PATH.exists():
        pytest.skip("no working database")
    s = server.serve(port=0, block=False)
    yield s
    s.shutdown(); s.server_close()


def _get(srv, path):
    port = srv.server_address[1]
    path = urllib.parse.quote(path, safe="/=,?&")  # the path may contain non-ASCII characters (group/dataset name)
    try:
        with urllib.request.urlopen("http://127.0.0.1:%d%s" % (port, path)) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def test_populations_and_big_graph_requires_focus(srv):
    st, body = _get(srv, "/api/populations/worm_cook2019")
    assert st == 200 and body["groups"]["backward_cmd"] == 6
    st, body = _get(srv, "/api/populations/missing")
    assert st == 404
    if not (paths.CACHE / "fafb_783.npz").exists():
        pytest.skip("no fafb_783")
    st, body = _get(srv, "/api/graph/fafb_783")
    assert st == 400 and "focus" in body["error"]
    st, body = _get(srv, "/api/graph/fafb_783?focus=mn9&top=50")
    assert st == 200 and 2 <= len(body["nodes"]) <= 52 and any(n["focus"] for n in body["nodes"])
    assert all("x" in n and "type" in n for n in body["nodes"])


def test_run_rates_by_names(srv):
    st, runs = _get(srv, "/api/runs")
    if not runs:
        pytest.skip("no runs")
    rid = runs[0]["id"]
    st, full = _get(srv, "/api/run/%s" % rid)
    names = full["names"][:3]
    st, part = _get(srv, "/api/run/%s?names=%s,zzz" % (rid, ",".join(names)))
    assert st == 200 and part["names"] == names and len(part["rates"][0]) == 3
    assert "extra" in part


def test_big_run_without_names_requires_names(srv):
    st, runs = _get(srv, "/api/runs")
    big_run = next((r for r in runs if r["dataset"] == "fafb_783"), None)
    if big_run is None:
        pytest.skip("no fafb_783 runs")
    st, body = _get(srv, "/api/run/%s" % big_run["id"])
    assert st == 400 and "names" in body["error"]


def test_page_fly_subgraph_and_state(srv):
    if not (paths.CACHE / "fafb_783.npz").exists():
        pytest.skip("no fafb_783")
    port = srv.server_address[1]
    out = paths.RESULTS / "screens"
    out.mkdir(parents=True, exist_ok=True)
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.goto("http://127.0.0.1:%d/" % port)
        page.wait_for_function("document.getElementById('status').textContent.includes('cells')", timeout=30000)

        dataset_ids = page.eval_on_selector_all("#dataset option", "els => els.map(e => e.value)")
        if "fafb_783" not in dataset_ids:
            browser.close()
            pytest.skip("fafb_783 is not in the page's dataset list")

        page.select_option("#dataset", "fafb_783")
        page.wait_for_function(
            "document.getElementById('status').textContent.includes('subgraph')", timeout=30000)

        group_ids = page.eval_on_selector_all("#group option", "els => els.map(e => e.value)")
        if "mn9" in group_ids:
            page.select_option("#group", "mn9")
            page.wait_for_function(
                "document.getElementById('status').textContent.includes('subgraph')", timeout=30000)
        status = page.text_content("#status")
        assert "fafb_783" in status and "subgraph" in status

        painted = page.evaluate("""() => { const c = document.getElementById('c'); const d = c.getContext('2d').getImageData(0,0,c.width,c.height).data;
            let n = 0; for (let i = 0; i < d.length; i += 4) if (d[i] > 40 || d[i+1] > 40 || d[i+2] > 40) n++; return n; }""")
        assert painted > 0

        fly_runs = page.eval_on_selector_all(
            "#run option", "els => els.filter(e => e.dataset.ds === 'fafb_783').map(e => e.value)")
        if fly_runs:
            page.select_option("#run", fly_runs[0])
            page.wait_for_function(
                "document.getElementById('status').textContent.includes('run')", timeout=15000)
            state_painted = page.evaluate("""() => { const c = document.getElementById('state'); const d = c.getContext('2d').getImageData(0,0,c.width,c.height).data;
                let n = 0; for (let i = 0; i < d.length; i += 4) if (d[i] > 20 || d[i+1] > 20 || d[i+2] > 20) n++; return n; }""")
            assert state_painted > 0

        page.screenshot(path=str(out / "view_fly.png"))
        browser.close()
    assert (out / "view_fly.png").stat().st_size > 10000
