"""Graded model for the worm: cells do not spike, activity r ∈ [0, 1].

Assumptions:
- dv/dt = (−v + gain·(W_chem·(r·x)) + g_gap·(W_gap·v − deg·v) + I_stim + noise)/τ;
- r = σ((v − θ)/w); x is the synaptic resource (Tsodyks–Markram), x ≡ 1 when std=None;
- noise is Gaussian, amplitude `noise` from the stimulus, a fixed seed.
All numbers are assumptions, not measurements; the gates check them.
"""
import numpy as np

from . import flags, result

MODEL = "graded"


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class Graded:
    def __init__(self, graph, dt_ms=1.0, tau_ms=50.0, gain=0.02, g_gap=0.01, theta=0.5, width=0.15, std=None, seed=0):
        self.g = graph
        self.params = {"dt_ms": dt_ms, "tau_ms": tau_ms, "gain": gain, "g_gap": g_gap, "theta": theta,
                       "width": width, "std": std}
        self.seed = seed
        self.W = graph.W_chem.astype(np.float32)
        self.G = graph.W_gap.astype(np.float32)
        self.deg = np.asarray(self.G.sum(axis=1)).ravel().astype(np.float32)

    def run(self, stimulus, duration_ms, window_ms=50.0):
        p = self.params
        rng = np.random.default_rng(self.seed)
        n = self.g.n
        v = np.zeros(n, np.float32)
        x = np.ones(n, np.float32)
        steps = int(round(duration_ms / p["dt_ms"]))
        per_win = max(1, int(round(window_ms / p["dt_ms"])))
        windows = int(np.ceil(steps / per_win))
        rates = np.zeros((windows, n), np.float32)
        std = p["std"]
        for k in range(steps):
            t = k * p["dt_ms"]
            r = _sigmoid((v - p["theta"]) / p["width"]).astype(np.float32)
            out = r * x if std else r
            syn = p["gain"] * (self.W @ out)
            gap = p["g_gap"] * (self.G @ v - self.deg * v)
            noise = rng.normal(0.0, stimulus.noise, n).astype(np.float32) if stimulus.noise else 0.0
            dv = (-v + syn + gap + stimulus.drive(self.g, t, p["dt_ms"]) + noise) / p["tau_ms"]
            v = v + p["dt_ms"] * dv
            if std:
                x = x + p["dt_ms"] * ((1.0 - x) / std["tau_rec_ms"] - std["u"] * r * x)
                x = np.clip(x, 0.0, 1.0)
            rates[k // per_win] += r
        # divide each window by the actual number of steps it contains (the last window can be shorter
        # than per_win if duration_ms is not a multiple of window_ms), not always by per_win.
        counts = np.full(windows, per_win, np.float32)
        if steps % per_win:
            counts[-1] = steps % per_win
        rates /= counts[:, None]
        fl = flags.check(rates, window_ms)
        return result.RunResult(self.g.dataset, MODEL, dict(p), self.seed, stimulus.to_dict(), p["dt_ms"], window_ms,
                                list(self.g.names), rates, fl, result.code_hash(), dataset_version=self.g.version,
                                dataset_params=dict(self.g.params))
