"""Stimulus: a list of pulses "these cells, this value, in this window".
kind="current": the value is in model units (graded: fractions of threshold; LIF: mV per ms added to voltage).
kind="poisson": the value is a rate, Hz; each cell gets an independent Poisson input
(LIF: an event adds w_syn·f_poi to v, as PoissonInput(target_var='v') does in Shiu 2024 — a jump
in voltage, not in the synaptic variable g; this is the authors' mechanism, not our assumption).
The graded model does not support kind="poisson" and raises ValueError."""
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class Pulse:
    names: list
    value: float
    t0_ms: float
    t1_ms: float
    kind: str = "current"


@dataclass
class CompiledPulse:
    idx: np.ndarray
    value: float
    k0: int
    k1: int
    kind: str


@dataclass
class Stimulus:
    pulses: list = field(default_factory=list)
    noise: float = 0.0

    def to_dict(self):
        return {"pulses": [asdict(p) for p in self.pulses], "noise": self.noise}

    @classmethod
    def from_dict(cls, d):
        return cls([Pulse(**p) for p in d.get("pulses", [])], float(d.get("noise", 0.0)))

    def compile(self, graph, dt_ms):
        """Cell indices and window boundaries in steps — computed once per run, not on every step."""
        return [CompiledPulse(graph.idx(p.names), float(p.value), int(round(p.t0_ms / dt_ms)),
                              int(round(p.t1_ms / dt_ms)), p.kind) for p in self.pulses]

    def drive(self, graph, t_ms, dt_ms=1.0):
        """Current at step t_ms (for the graded model). Poisson pulses do not take part here."""
        key = (id(graph), graph.dataset, graph.n, dt_ms)
        if getattr(self, "_compiled_key", None) != key:
            self._compiled = self.compile(graph, dt_ms)
            self._compiled_key = key
        d = np.zeros(graph.n, dtype=np.float32)
        k = int(round(t_ms / dt_ms))
        for c in self._compiled:
            if c.kind == "poisson":
                raise ValueError("kind='poisson' is supported only by LIF")
            if c.k0 <= k < c.k1:
                d[c.idx] += c.value
        return d
