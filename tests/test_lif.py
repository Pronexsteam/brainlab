import numpy as np

from brainlab.sim import lif, stimulus
from brainlab.store import db, graph
from brainlab.store.loaders import worm_cook2019 as worm


def _graph(tmp_db, worm_raw):
    conn = db.connect(tmp_db); worm.load(conn, worm_raw)
    return graph.build(conn, worm.DATASET_ID)


def test_lif_deterministic_and_spikes(tmp_db, worm_raw):
    g = _graph(tmp_db, worm_raw)
    # 5.0 mV/ms (used to be 2.0 with the old LIF form): in the Shiu form, gap junctions to resting neighbors
    # (deg 35-46 for ALM/AVM) add extra leak g_gap·deg, so 2 mV/ms falls short of the threshold.
    st = stimulus.Stimulus([stimulus.Pulse(["ALML", "ALMR", "AVM"], 5.0, 100, 400)])
    a = lif.LIF(g, seed=1, device="cpu").run(st, 500, window_ms=50)
    b = lif.LIF(g, seed=1, device="cpu").run(st, 500, window_ms=50)
    assert np.array_equal(a.rates, b.rates)
    assert a.rates.shape == (10, g.n) and a.rates.max() <= 1.0
    assert a.group_rate(["ALML", "ALMR", "AVM"], 150, 400) > 0.05     # stimulated cells spike
    assert a.group_rate(["ALML", "ALMR", "AVM"], 0, 100) == 0.0        # silence before the stimulus
    assert a.model == "lif" and "w_syn" in a.params


def test_lif_seizure_at_rate_ceiling(tmp_db, worm_raw):
    """A huge constant current on all cells: each spikes at the limit allowed by
    refractoriness (once every round(refractory_ms/dt) steps, ISI >= refractory_ms, as in Brian2
    >= 2.1); normalized rates should be ~1.0, not <=0.333, and flags.check must see a seizure."""
    g = _graph(tmp_db, worm_raw)
    st = stimulus.Stimulus([stimulus.Pulse(list(g.names), 1000.0, 0, 1000)])
    r = lif.LIF(g, seed=0, device="cpu").run(st, 1000, window_ms=100)
    assert r.rates[-1].mean() > 0.9
    assert r.flags["seizure"]


def test_lif_std_reduces_output():
    """This test used to be vacuous (0 <= 0, PLM has no chemical output onto PVC at the fly's
    calibration w_syn=0.275). Replaced with the toy graph _chain(400) from test_lif_shiu (a->b, 400 synapses,
    w_syn·400 = 110 mV into g -> b spikes): 5 current pulses onto a; with depression (std), b's rate on the
    last pulse is strictly less than on the first, without depression — not less."""
    from tests.test_lif_shiu import _chain
    g = _chain(400)
    pulses = [stimulus.Pulse(["a"], 200.0, 100 + i * 500, 102 + i * 500, kind="current") for i in range(5)]
    st = stimulus.Stimulus(pulses)
    plain = lif.LIF(g, seed=0, device="cpu").run(st, 2600, window_ms=50)
    dep = lif.LIF(g, seed=0, std={"u": 0.5, "tau_rec_ms": 2000}, device="cpu").run(st, 2600, window_ms=50)
    first_plain = plain.group_rate(["b"], 100, 150)
    last_plain = plain.group_rate(["b"], 2100, 2150)
    first_dep = dep.group_rate(["b"], 100, 150)
    last_dep = dep.group_rate(["b"], 2100, 2150)
    assert last_dep < first_dep
    assert last_plain >= first_plain
