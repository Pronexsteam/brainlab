import numpy as np
import scipy.sparse as sp
import torch

from brainlab.sim import lif, stimulus
from brainlab.store.graph import Graph


def _chain(w):
    """a -> b with weight w (synapse count), no gap junctions."""
    Wc = sp.csr_matrix((np.array([w], np.float32), ([1], [0])), shape=(2, 2))
    return Graph("toy", ["a", "b"], ["neuron"] * 2, ["", ""], Wc, sp.csr_matrix((2, 2), dtype=np.float32), version="t")


def test_poisson_drives_spikes_near_rate():
    g = _chain(0)
    st = stimulus.Stimulus([stimulus.Pulse(["a"], 150.0, 0, 4000, kind="poisson")])
    r = lif.LIF(g, seed=3, device="cpu").run(st, 4000, window_ms=4000)
    hz = r.hz(["a"])
    assert 130 < hz < 170, hz               # the authors' mechanism: a jump in v, refractoriness 0 -> exactly one spike per event,
                                            # rate = r_poi 150 Hz +/- Poisson spread (4 s: sigma ~= 6 Hz; seed 3 -> 154.5)
    assert r.hz(["b"]) == 0.0 and r.extra["form"] == "shiu2024" and 400 < r.extra["max_rate_hz"] < 500   # 1000/round(2.2/0.1) = 1000/22 ~= 454.5 Hz


def test_single_synapse_epsp_matches_shiu_scale():
    """One spike through 1 synapse: g += 0.275 mV, v rises by no more than w·τ_syn/τ_m ~= 0.07 mV.
    In the old form the increment was ~20x larger."""
    g = _chain(1)
    st = stimulus.Stimulus([stimulus.Pulse(["a"], 200.0, 0, 0.2, kind="current")])   # 2 steps of 20 mV: a spikes exactly once
    m = lif.LIF(g, seed=0, device="cpu")
    r = m.run(st, 20, window_ms=20)
    assert r.hz(["a"]) > 0 and r.hz(["b"]) == 0
    assert 0 < m.v_peak_b < 0.1                # see step 3: debug field m.v_peak_b, max(v_b) − v_rest


def test_strong_synapse_spikes_with_delay():
    g = _chain(400)                             # 400·0.275 = 110 mV into g -> b spikes
    st = stimulus.Stimulus([stimulus.Pulse(["a"], 200.0, 0, 2, kind="current")])
    r = lif.LIF(g, seed=0, device="cpu").run(st, 10, window_ms=1)
    ka = int(np.argmax(r.rates[:, 0] > 0)); kb = int(np.argmax(r.rates[:, 1] > 0))
    assert kb - ka >= 1                         # delay 1.8 ms -> b is later than a by at least one window


def test_bitwise_repeat_cpu():
    g = _chain(400)
    st = stimulus.Stimulus([stimulus.Pulse(["a"], 100.0, 0, 500, kind="poisson")], noise=0.2)
    a = lif.LIF(g, seed=7, device="cpu").run(st, 500)
    b = lif.LIF(g, seed=7, device="cpu").run(st, 500)
    assert np.array_equal(a.rates, b.rates)


def test_repeat_cuda_recorded():
    if not torch.cuda.is_available():
        import pytest; pytest.skip("no CUDA")
    g = _chain(400)
    st = stimulus.Stimulus([stimulus.Pulse(["a"], 100.0, 0, 500, kind="poisson")], noise=0.2)
    a = lif.LIF(g, seed=7).run(st, 500); b = lif.LIF(g, seed=7).run(st, 500)
    assert a.extra["device"].startswith("cuda") and np.array_equal(a.rates, b.rates)
