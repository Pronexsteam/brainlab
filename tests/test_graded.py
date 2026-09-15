import numpy as np

from brainlab import paths
from brainlab.sim import graded, result, stimulus
from brainlab.store import db, graph
from brainlab.store.loaders import worm_cook2019 as worm


def _graph(tmp_db, worm_raw):
    conn = db.connect(tmp_db); worm.load(conn, worm_raw)
    return graph.build(conn, worm.DATASET_ID)


def test_deterministic_and_bounded(tmp_db, worm_raw):
    g = _graph(tmp_db, worm_raw)
    st = stimulus.Stimulus([stimulus.Pulse(["ALML", "ALMR", "AVM"], 1.0, 200, 400)], noise=0.02)
    a = graded.Graded(g, seed=3).run(st, 600, window_ms=50)
    b = graded.Graded(g, seed=3).run(st, 600, window_ms=50)
    assert np.array_equal(a.rates, b.rates)                      # то же зерно — те же числа
    c = graded.Graded(g, seed=4).run(st, 600, window_ms=50)
    assert not np.array_equal(a.rates, c.rates)                  # другое зерно — другой шум
    assert a.rates.shape == (12, g.n) and a.rates.min() >= 0 and a.rates.max() <= 1
    assert a.model == "graded" and a.dataset == worm.DATASET_ID and a.stimulus == st.to_dict()
    assert not a.flags["silent"] and not a.flags["seizure"]


def test_stimulus_moves_target(tmp_db, worm_raw):
    g = _graph(tmp_db, worm_raw)
    st = stimulus.Stimulus([stimulus.Pulse(["ALML", "ALMR"], 1.0, 200, 400)])
    r = graded.Graded(g, seed=0).run(st, 500, window_ms=50)
    before = r.group_rate(["ALML", "ALMR"], 0, 200)
    during = r.group_rate(["ALML", "ALMR"], 250, 400)
    assert during > before + 0.2


def test_run_dataset_version_roundtrip(tmp_db, worm_raw, tmp_path, monkeypatch):
    """RunResult.dataset_version заполняется из graph.version и переживает save/load."""
    monkeypatch.setattr(paths, "RESULTS", tmp_path)
    conn = db.connect(tmp_db); worm.load(conn, worm_raw)
    g = graph.build(conn, worm.DATASET_ID)
    info = db.dataset_info(conn, worm.DATASET_ID)
    st = stimulus.Stimulus([stimulus.Pulse(["ALML", "ALMR"], 1.0, 0, 50)])
    r = graded.Graded(g, seed=0).run(st, 100, window_ms=50)
    assert r.dataset_version == info["version"] and info["version"]
    folder = r.save()
    r2 = result.load(folder)
    assert r2.dataset_version == info["version"]


def test_run_dataset_params_roundtrip(tmp_db, worm_raw, tmp_path, monkeypatch):
    """RunResult.dataset_params заполняется из graph.params и переживает save/load (задача 1
    финальной волны): у червя там sign_rule."""
    monkeypatch.setattr(paths, "RESULTS", tmp_path)
    conn = db.connect(tmp_db); worm.load(conn, worm_raw)
    g = graph.build(conn, worm.DATASET_ID)
    st = stimulus.Stimulus([stimulus.Pulse(["ALML", "ALMR"], 1.0, 0, 50)])
    r = graded.Graded(g, seed=0).run(st, 100, window_ms=50)
    assert "sign_rule" in r.dataset_params and r.dataset_params["sign_rule"]
    folder = r.save()
    r2 = result.load(folder)
    assert r2.dataset_params == r.dataset_params


def test_last_window_not_underaveraged(tmp_db, worm_raw):
    """duration_ms не кратен window_ms → последнее (неполное) окно делится на фактическое
    число шагов в нём, а не на per_win, поэтому при постоянной активности не меньше предыдущих."""
    g = _graph(tmp_db, worm_raw)
    st = stimulus.Stimulus([stimulus.Pulse(["ALML", "ALMR", "AVM"], 5.0, 0, 1000)], noise=0.0)
    r = graded.Graded(g, seed=0).run(st, 430, window_ms=100)     # 430 = 4*100 + 30
    assert r.rates.shape[0] == 5
    prev = r.group_rate(["ALML", "ALMR", "AVM"], 300, 400)
    last = r.group_rate(["ALML", "ALMR", "AVM"], 400, 430)
    assert last >= prev - 1e-3


def test_std_depresses_repeated_input(tmp_db, worm_raw):
    g = _graph(tmp_db, worm_raw)
    pulses = [stimulus.Pulse(["ALML", "ALMR"], 1.0, 200 + i * 1000, 400 + i * 1000) for i in range(6)]
    st = stimulus.Stimulus(pulses)
    # PLM в этом наборе не имеет вообще ни одного химического выхода (только щелевые контакты на
    # PVC/LUA/PHC), поэтому источником берём передний тактильный рецептор ALM — у него есть химические
    # выходы (например, на BDUL); сильнейшую химическую цель берём из графа, а не предполагаем заранее.
    j = g.idx(["ALML"])[0]
    col = g.W_chem[:, j].toarray().ravel()
    i = int(col.argmax())
    assert col[i] > 0
    target = [g.names[i]]
    with_std = graded.Graded(g, seed=0, std={"u": 0.08, "tau_rec_ms": 480}).run(st, 6400, window_ms=50)
    without = graded.Graded(g, seed=0).run(st, 6400, window_ms=50)
    baseline = with_std.group_rate(target, 0, 200)
    first = with_std.group_rate(target, 250, 400)
    last = with_std.group_rate(target, 5250, 5400)
    assert first > baseline + 0.01                                # стимул вообще доходит
    assert last < first                                           # с депрессией ответ на повтор слабеет
    f0, l0 = without.group_rate(target, 250, 400), without.group_rate(target, 5250, 5400)
    assert l0 >= f0 - 1e-3  # без депрессии ответ на повтор не слабеет (здесь он даже растёт — накопление в возвратной сети при этих константах)
