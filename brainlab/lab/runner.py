"""An experiment from YAML → run → evaluation. The experiment file is the only input, the code does not change the run."""
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
            raise ValueError("experiment is missing field %s" % key)
    spec.setdefault("params", {}); spec.setdefault("seed", 0); spec.setdefault("window_ms", 50.0)
    spec.setdefault("extreme", False)
    return spec


def _expand_groups(spec):
    """Expands stimulus.pulses[].names of the form "group:<name>" into a list from populations.
    Returns (the expanded stimulus dict, {pulse index: original "group:<name>"}, [empty groups]).
    An empty group (no labels in the dataset) does not fail the run: the pulse is skipped, the group's
    name is added to the list of empty groups."""
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
    g = graph_mod.get(spec["dataset"], normalize=gspec.get("normalize"))   # graph: {normalize: fafb_783} — input normalization
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
    # first ≈ 0 — there is no response at all to the first stimulus (not "got worse"/"got better", but a
    # defective run: nothing to compare against); previously ratio=inf gave valence=-1 via pain_if_above
    # (no response looked like pain), now valence=0 and an honest no_response flag.
    ratio = last / first if not no_response else None      # None, not inf: json/JS do not know Infinity
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
        valence = -1                                   # a defective run is always pain
    out["valence"] = valence
    return out
