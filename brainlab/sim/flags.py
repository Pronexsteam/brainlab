"""Run defects: silence (the network stays quiet) and seizure (nearly all cells pinned at the ceiling
the whole time). Thresholds are in fractions of the maximum (SEIZURE_LEVEL=0.9 etc.), so they are the
same for the graded model (r ∈ [0, 1] is already the ceiling) and for LIF: LIF normalizes rates by
max_rate = 1 / round(refractory_ms/dt) — the rate ceiling allowed by refractoriness (as in Brian2),
so 1.0 means "at the ceiling" for both models (see the lif.py docstring)."""
import numpy as np

SILENT_MAX = 1e-4
SEIZURE_FRACTION = 0.8
SEIZURE_LEVEL = 0.9


def check(rates, window_ms):
    r = np.asarray(rates, dtype=np.float32)
    mean = float(r.mean()) if r.size else 0.0
    silent = bool(r.max() <= SILENT_MAX) if r.size else True
    high = (r >= SEIZURE_LEVEL).mean(axis=1) if r.size else np.zeros(0)
    seizure = bool(high.size and (high >= SEIZURE_FRACTION).mean() >= 0.5)
    return {"silent": silent, "seizure": seizure, "mean_rate": mean}
