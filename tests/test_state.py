import numpy as np
import pytest

from brainlab.lab import state
from brainlab.sim import result


def _run():
    rates = np.array([[0.0, 1.0, 0.5], [0.2, 0.4, 0.6]], np.float32)
    return result.RunResult("toy", "graded", {}, 0, {}, 1.0, 50.0, ["a", "b", "c"], rates, {}, "h", extra={"max_rate_hz": 100.0})


def test_vector_and_attach():
    # float32 storage of rates (as in RunResult) does not give the decimal 0.3/0.6 an exact float64 bit
    # pattern, so the comparison uses pytest.approx, not ==; the value itself (the group mean) is correct.
    r = _run()
    v = state.vector(r, {"ab": ["a", "b"], "none": [], "c": ["c", "zzz"]})
    assert v["ab"] == pytest.approx([0.5, 0.3]) and v["none"] == [0.0, 0.0] and v["c"] == pytest.approx([0.5, 0.6])
    state.attach(r, groups={"ab": ["a", "b"]})
    assert r.extra["state"]["values"]["ab"] == pytest.approx([0.5, 0.3]) and r.extra["state"]["groups"] == ["ab"]
    assert r.extra["state_hz"]["ab"] == pytest.approx([50.0, 30.0])
