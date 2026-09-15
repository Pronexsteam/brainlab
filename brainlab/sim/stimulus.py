"""Стимул: список импульсов «эти клетки, такое значение, в таком окне».
kind="current": значение — единицы модели (плавная: доли порога; LIF: мВ за мс к напряжению).
kind="poisson": значение — частота, Гц; каждая клетка получает независимый пуассоновский вход
(LIF: событие прибавляет w_syn·f_poi к v, как PoissonInput(target_var='v') у Shiu 2024 — скачок
в напряжение, а не в синаптическую переменную g; это авторский механизм, не наше допущение).
Плавная модель kind="poisson" не поддерживает и бросает ValueError."""
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
        """Индексы клеток и границы окон в шагах — один раз на прогон, не на каждом шаге."""
        return [CompiledPulse(graph.idx(p.names), float(p.value), int(round(p.t0_ms / dt_ms)),
                              int(round(p.t1_ms / dt_ms)), p.kind) for p in self.pulses]

    def drive(self, graph, t_ms, dt_ms=1.0):
        """Ток на шаге t_ms (для плавной модели). Пуассоновские импульсы здесь не участвуют."""
        key = (id(graph), graph.dataset, graph.n, dt_ms)
        if getattr(self, "_compiled_key", None) != key:
            self._compiled = self.compile(graph, dt_ms)
            self._compiled_key = key
        d = np.zeros(graph.n, dtype=np.float32)
        k = int(round(t_ms / dt_ms))
        for c in self._compiled:
            if c.kind == "poisson":
                raise ValueError("kind='poisson' поддерживает только LIF")
            if c.k0 <= k < c.k1:
                d[c.idx] += c.value
        return d
