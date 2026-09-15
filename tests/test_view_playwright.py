import pytest

from brainlab import paths
from brainlab.view import server

playwright = pytest.importorskip("playwright.sync_api")


@pytest.fixture(scope="module")
def srv():
    if not paths.DB_PATH.exists():
        pytest.skip("нет рабочей базы")
    s = server.serve(port=0, block=False)
    yield s
    s.shutdown()
    s.server_close()


def test_page_draws_graph_and_plays(srv):
    port = srv.server_address[1]
    out = paths.RESULTS / "screens"
    out.mkdir(parents=True, exist_ok=True)
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.goto("http://127.0.0.1:%d/" % port)
        page.wait_for_function("document.getElementById('status').textContent.includes('клеток')", timeout=30000)
        status = page.text_content("#status")
        assert "worm_cook2019" in status
        # холст не пустой: есть пиксели, отличные от фона
        painted = page.evaluate("""() => { const c = document.getElementById('c'); const d = c.getContext('2d').getImageData(0,0,c.width,c.height).data;
            let n = 0; for (let i = 0; i < d.length; i += 4) if (d[i] > 40 || d[i+1] > 40 || d[i+2] > 40) n++; return n; }""")
        assert painted > 500
        page.click("#c", position={"x": 640, "y": 400})
        runs = page.eval_on_selector_all("#run option", "els => els.map(e => e.value).filter(Boolean)")
        if runs:
            page.select_option("#run", runs[0])
            page.wait_for_function("document.getElementById('status').textContent.includes('прогон')", timeout=15000)
            page.click("#play")
            page.wait_for_timeout(500)
            assert int(page.input_value("#time")) > 0
        page.screenshot(path=str(out / "view_worm.png"))
        browser.close()
    assert (out / "view_worm.png").stat().st_size > 10000
