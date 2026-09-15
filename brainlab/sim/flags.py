"""Брак прогона: тишина (сеть молчит) и судорога (почти все клетки на потолке всё время).
Пороги — на долях максимума (SEIZURE_LEVEL=0.9 и т.д.), поэтому одинаковы для плавной
модели (r ∈ [0, 1] уже предел) и для LIF: LIF нормирует rates на max_rate =
1 / round(refractory_ms/dt) — предельную частоту, допустимую рефрактерностью (как у Brian2),
так что 1.0 у обеих моделей значит «на пределе» (см. докстринг lif.py)."""
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
