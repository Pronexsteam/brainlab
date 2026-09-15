"""Вектор состояния прогона: средняя активность подписанных групп в каждом окне (дизайн §7а).
Слово «эмоции» не используем: это показания групп клеток, не больше."""
import numpy as np

from . import populations


def vector(run, groups):
    pos = {n: i for i, n in enumerate(run.names)}
    out = {}
    for g, names in groups.items():
        idx = [pos[n] for n in names if n in pos]
        out[g] = [float(x) for x in run.rates[:, idx].mean(axis=1)] if idx else [0.0] * run.rates.shape[0]
    return out


def attach(run, dataset_id=None, groups=None, names=None):
    if groups is None:
        allg = populations.resolve(dataset_id)
        keys = names or populations.state_groups(dataset_id)
        groups = {k: allg[k] for k in keys}
    v = vector(run, groups)
    run.extra["state"] = {"groups": list(groups), "windows": int(run.rates.shape[0]), "values": v}
    if "max_rate_hz" in run.extra:
        k = float(run.extra["max_rate_hz"])
        run.extra["state_hz"] = {g: [x * k for x in vals] for g, vals in v.items()}
    return run.extra["state"]
