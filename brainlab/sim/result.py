"""A run's result and its description. Folder results/<id>/: run.json (everything it was produced from),
rates.npz (windows × cells), summary.md. Runs are never deleted; there is an archived flag."""
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .. import paths


def code_hash():
    h = hashlib.sha256()
    for p in sorted((paths.ROOT / "brainlab").rglob("*.py")):
        h.update(p.relative_to(paths.ROOT).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def new_run_id(base=None):
    base = Path(base) if base else paths.RESULTS
    base.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    k = 0
    while (base / ("%s-%02d" % (stamp, k))).exists():
        k += 1
    return "%s-%02d" % (stamp, k)


@dataclass
class RunResult:
    dataset: str
    model: str
    params: dict
    seed: int
    stimulus: dict
    dt_ms: float
    window_ms: float
    names: list
    rates: np.ndarray
    flags: dict
    code_hash: str
    valence: int = 0
    parent: str = ""
    archived: bool = False
    id: str = ""
    extra: dict = field(default_factory=dict)
    dataset_version: str = ""          # dataset version (graph.version); empty — old run.json files predating task 7
    dataset_params: dict = field(default_factory=dict)   # dataset params (graph.params: sign_rule/min_count/...); empty — old run.json files predating the final wave

    def group_rate(self, names, t0_ms=None, t1_ms=None):
        pos = {n: i for i, n in enumerate(self.names)}
        idx = [pos[n] for n in names if n in pos]
        w0 = 0 if t0_ms is None else int(t0_ms // self.window_ms)
        w1 = self.rates.shape[0] if t1_ms is None else int(np.ceil(t1_ms / self.window_ms))
        return float(self.rates[w0:w1][:, idx].mean()) if idx else 0.0

    def hz(self, names, t0_ms=None, t1_ms=None):
        """Mean firing rate of the group in Hz (LIF); the graded model has no extra["max_rate_hz"] → ValueError."""
        if "max_rate_hz" not in self.extra:
            raise ValueError("hz() is for LIF only: extra has no max_rate_hz")
        return self.group_rate(names, t0_ms, t1_ms) * float(self.extra["max_rate_hz"])

    def rates_hz(self):
        """rates in Hz (LIF): normalized rates · max_rate_hz."""
        if "max_rate_hz" not in self.extra:
            raise ValueError("rates_hz() is for LIF only: extra has no max_rate_hz")
        return self.rates * float(self.extra["max_rate_hz"])

    def save(self, base=None):
        base = Path(base) if base else paths.RESULTS
        self.id = self.id or new_run_id(base)
        folder = base / self.id
        folder.mkdir(parents=True, exist_ok=True)
        meta = asdict(self)
        meta.pop("rates")
        (folder / "run.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        np.savez_compressed(folder / "rates.npz", rates=self.rates.astype(np.float32))
        dp_line = ""
        if self.dataset_params.get("sign_rule") or self.dataset_params.get("min_count"):
            dp_line = "\nsign rule: %s, min_count: %s\n" % (
                self.dataset_params.get("sign_rule", "-"), self.dataset_params.get("min_count", "-"))
        (folder / "summary.md").write_text(
            "# Run %s\n\ndataset %s (version %s), model %s, seed %d, %d windows of %.0f ms, %d cells\n%s\n"
            "flags: %s\n\nvalence: %+d, parent: %s\n" % (
                self.id, self.dataset, self.dataset_version or "-", self.model, self.seed, self.rates.shape[0],
                self.window_ms, self.rates.shape[1], dp_line, json.dumps(self.flags, ensure_ascii=False), self.valence,
                self.parent or "-"),
            encoding="utf-8")
        return folder


def load(path):
    path = Path(path)
    meta = json.loads((path / "run.json").read_text(encoding="utf-8"))
    rates = np.load(path / "rates.npz")["rates"]
    return RunResult(rates=rates, **meta)


def list_runs(base=None):
    base = Path(base) if base else paths.RESULTS
    out = []
    for folder in sorted(base.glob("*")) if base.exists() else []:
        f = folder / "run.json"
        if f.exists():
            m = json.loads(f.read_text(encoding="utf-8"))
            out.append({"id": folder.name, "dataset": m["dataset"], "model": m["model"], "seed": m["seed"],
                        "valence": m.get("valence", 0), "archived": m.get("archived", False), "flags": m["flags"]})
    return out
