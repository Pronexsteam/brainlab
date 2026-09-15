import numpy as np
import scipy.sparse as sp
import torch

from brainlab.sim import lif, stimulus
from brainlab.store.graph import Graph


def _chain(w):
    """a → b с весом w (число синапсов), без щелей."""
    Wc = sp.csr_matrix((np.array([w], np.float32), ([1], [0])), shape=(2, 2))
    return Graph("toy", ["a", "b"], ["neuron"] * 2, ["", ""], Wc, sp.csr_matrix((2, 2), dtype=np.float32), version="t")


def test_poisson_drives_spikes_near_rate():
    g = _chain(0)
    st = stimulus.Stimulus([stimulus.Pulse(["a"], 150.0, 0, 4000, kind="poisson")])
    r = lif.LIF(g, seed=3, device="cpu").run(st, 4000, window_ms=4000)
    hz = r.hz(["a"])
    assert 130 < hz < 170, hz               # механизм авторов: скачок в v, рефрактерность 0 → ровно один спайк на событие,
                                            # частота = r_poi 150 Гц ± пуассоновский разброс (4 с: σ ≈ 6 Гц; seed 3 → 154,5)
    assert r.hz(["b"]) == 0.0 and r.extra["form"] == "shiu2024" and 400 < r.extra["max_rate_hz"] < 500   # 1000/round(2.2/0.1) = 1000/22 ≈ 454,5 Гц


def test_single_synapse_epsp_matches_shiu_scale():
    """Один спайк через 1 синапс: g += 0.275 мВ, v растёт не больше чем на w·τ_syn/τ_m ≈ 0.07 мВ.
    В прежней форме прибавка была ~20 раз больше."""
    g = _chain(1)
    st = stimulus.Stimulus([stimulus.Pulse(["a"], 200.0, 0, 0.2, kind="current")])   # 2 шага по 20 мВ: a спайкует ровно один раз
    m = lif.LIF(g, seed=0, device="cpu")
    r = m.run(st, 20, window_ms=20)
    assert r.hz(["a"]) > 0 and r.hz(["b"]) == 0
    assert 0 < m.v_peak_b < 0.1                # см. шаг 3: отладочное поле m.v_peak_b, max(v_b) − v_rest


def test_strong_synapse_spikes_with_delay():
    g = _chain(400)                             # 400·0.275 = 110 мВ в g → b спайкует
    st = stimulus.Stimulus([stimulus.Pulse(["a"], 200.0, 0, 2, kind="current")])
    r = lif.LIF(g, seed=0, device="cpu").run(st, 10, window_ms=1)
    ka = int(np.argmax(r.rates[:, 0] > 0)); kb = int(np.argmax(r.rates[:, 1] > 0))
    assert kb - ka >= 1                         # задержка 1,8 мс → b позже a хотя бы на окно


def test_bitwise_repeat_cpu():
    g = _chain(400)
    st = stimulus.Stimulus([stimulus.Pulse(["a"], 100.0, 0, 500, kind="poisson")], noise=0.2)
    a = lif.LIF(g, seed=7, device="cpu").run(st, 500)
    b = lif.LIF(g, seed=7, device="cpu").run(st, 500)
    assert np.array_equal(a.rates, b.rates)


def test_repeat_cuda_recorded():
    if not torch.cuda.is_available():
        import pytest; pytest.skip("нет CUDA")
    g = _chain(400)
    st = stimulus.Stimulus([stimulus.Pulse(["a"], 100.0, 0, 500, kind="poisson")], noise=0.2)
    a = lif.LIF(g, seed=7).run(st, 500); b = lif.LIF(g, seed=7).run(st, 500)
    assert a.extra["device"].startswith("cuda") and np.array_equal(a.rates, b.rates)
