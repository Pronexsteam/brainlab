"""Ворота 1. Червь: переднее касание → «назад» сильнее «вперёд», заднее — наоборот.
Известно с 1985 года; здесь проверяется, что схема + наши предположения это дают."""
from ...sim import graded, stimulus
from ...store import graph as graph_mod
from .. import groups

DATASET = "worm_cook2019"


def _response(g, touch_names, seed):
    st = stimulus.Stimulus([stimulus.Pulse(touch_names, 1.0, 300, 600)], noise=0.0)
    r = graded.Graded(g, seed=seed).run(st, 800, window_ms=50)
    base_b = r.group_rate(groups.WORM["backward_cmd"], 0, 300)
    base_f = r.group_rate(groups.WORM["forward_cmd"], 0, 300)
    back = r.group_rate(groups.WORM["backward_cmd"], 350, 600) - base_b
    fwd = r.group_rate(groups.WORM["forward_cmd"], 350, 600) - base_f
    return back, fwd, r


def run(graph=None, seed=0, save=True):
    g = graph or graph_mod.get(DATASET)
    ab, af, r1 = _response(g, groups.WORM["anterior_touch"], seed)
    pb, pf, r2 = _response(g, groups.WORM["posterior_touch"], seed)
    ids = []
    for r, label in ((r1, "gate1_anterior"), (r2, "gate1_posterior")):
        r.extra["gate"] = label
        if save:
            ids.append(r.save().name)
    out = {"anterior_back": ab, "anterior_fwd": af, "posterior_back": pb, "posterior_fwd": pf,
           "passed": bool(ab > af and pf > pb), "run_ids": ids}
    return out


if __name__ == "__main__":
    print(run())
