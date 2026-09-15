"""Опыт из YAML → прогон → оценка. Файл опыта — единственный вход, код прогон не меняет."""
from pathlib import Path

import yaml

from ..sim import graded, lif, stimulus
from ..store import graph as graph_mod
from . import populations, state as state_mod

MODELS = {"graded": graded.Graded, "lif": lif.LIF}


def load_experiment(path):
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key in ("name", "dataset", "model", "duration_ms", "stimulus"):
        if key not in spec:
            raise ValueError("в опыте нет поля %s" % key)
    spec.setdefault("params", {}); spec.setdefault("seed", 0); spec.setdefault("window_ms", 50.0)
    spec.setdefault("extreme", False)
    return spec


def _expand_groups(spec):
    """Разворачивает stimulus.pulses[].names вида "group:<имя>" в список из populations.
    Возвращает (развёрнутый stimulus-словарь, {индекс импульса: исходное "group:<имя>"}, [пустые группы]).
    Пустая группа (нет подписей в наборе) не роняет прогон: импульс пропускается, имя группы
    попадает в список пустых групп."""
    st = {"noise": spec["stimulus"].get("noise", 0.0), "pulses": []}
    used = {}
    empty = []
    for i, p in enumerate(spec["stimulus"]["pulses"]):
        p = dict(p)
        n = p["names"]
        if isinstance(n, str) and n.startswith("group:"):
            group = n[len("group:"):]
            names = populations.names(spec["dataset"], group)
            if not names:
                if group not in empty:
                    empty.append(group)
                continue
            used[i] = n
            p["names"] = names
        st["pulses"].append(p)
    return st, used, empty


def run_experiment(path, save=True):
    spec = load_experiment(path)
    gspec = spec.get("graph", {}) or {}
    g = graph_mod.get(spec["dataset"], normalize=gspec.get("normalize"))   # graph: {normalize: fafb_783} — нормировка входов
    model = MODELS[spec["model"]](g, seed=int(spec["seed"]), **spec["params"])
    st_dict, used_groups, empty_groups = _expand_groups(spec)
    st = stimulus.Stimulus.from_dict(st_dict)
    r = model.run(st, float(spec["duration_ms"]), float(spec["window_ms"]))
    r.extra["experiment"] = spec["name"]
    r.extra["extreme"] = bool(spec["extreme"])
    if gspec:
        r.extra["graph"] = gspec
    if used_groups:
        r.extra["experiment_groups"] = used_groups
    if empty_groups:
        r.extra["empty_groups"] = empty_groups
    ev = evaluate(r, spec)
    r.valence = ev["valence"]
    r.extra["evaluation"] = ev
    try:
        state_mod.attach(r, spec["dataset"], names=spec.get("state_groups"))
    except KeyError as err:
        r.extra["state_missing"] = str(err)
    if save:
        r.save()
    return r


def _names(group_key, dataset):
    return populations.names(dataset, group_key)


def evaluate(r, spec):
    v = spec.get("valence")
    if not v:
        return {"valence": 0}
    names = _names(v["group"], r.dataset)
    first = r.group_rate(names, *v["first"])
    last = r.group_rate(names, *v["last"])
    no_response = first < 1e-9
    # first ≈ 0 — на первый стимул нет ответа вообще (не «стало хуже»/«стало лучше», а брак
    # прогона: сравнивать не с чем); раньше ratio=inf давал valence=-1 через pain_if_above
    # (нет ответа выглядело как боль), теперь valence=0 и честный флаг no_response.
    ratio = last / first if not no_response else None      # None, не inf: json/JS не знают Infinity
    out = {"first": first, "last": last, "ratio": ratio}
    if "novel" in v:
        out["novel"] = r.group_rate(_names(v.get("novel_group", v["group"]), r.dataset), *v["novel"])
    if no_response:
        out["no_response"] = True
        valence = 0
    else:
        valence = 0
        if ratio < v.get("reward_if_below", -1):
            valence = 1
        elif ratio > v.get("pain_if_above", float("inf")):
            valence = -1
    if r.flags.get("silent") or r.flags.get("seizure"):
        valence = -1                                   # брак — всегда боль
    out["valence"] = valence
    return out
