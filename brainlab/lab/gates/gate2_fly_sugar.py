"""Gate 2. Fly: sugar receptors → proboscis motor neuron MN9; bitter suppresses the response.
Reference — the output of the Shiu et al. 2024 model (Brian2) on the same v630 data: sugarR.parquet
from their repository (21 right-side sugar cells, 150 Hz Poisson, 1 s, 30 repeats; MN9 93.3 ± 3.2 Hz).
Criteria: 0.6 ≤ MN9_ours/MN9_oracle ≤ 1.4; Spearman over the reference's 200 most active cells ≥ 0.7;
the number of active (>1 Hz) cells within 0.7-1.4 of the reference; sugar+bitter ≤ 0.5·sugar on MN9.
Our Poisson generator and Brian2's are different generators, so the comparison is by rates and ranking,
not by spikes.
Caveat: the shipped parquet's absolute rates (GRN 197 Hz, MN9 93 Hz) are not reproduced by the authors'
code on current Brian2 2.10 (150 / 83 Hz instead); the cell ranking IS reproduced (ρ 0.987 on the top-200).
Caveat about local Brian2 (authors' code, 82.7 Hz): ratio ≈ 1.35 against the criterion's 1.4 boundary —
our run is closer to the boundary relative to the locally reproduced reference than relative to the
shipped parquet (ratio ≈ 1.2); the criterion is evaluated only against the shipped parquet, the local
Brian2 run is an informational cross-check (see results/oracle_brian2/sugarR_ours.parquet, if present)."""
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ... import paths
from ...sim import lif, stimulus
from ...store import graph as graph_mod
from .. import populations

DATASET = "flywire_630_shiu"
RATE_HZ = 150.0
T_MS = 1000.0
MN9 = "720575940660219265"


def oracle_rates(raw_dir, filename="sugarR.parquet"):
    """id → Hz: average number of spikes per repeat from the reference parquet, divided by t_run = 1 s."""
    df = pd.read_parquet(raw_dir / filename, columns=["trial", "flywire_id"])
    n_run = int(df["trial"].nunique())
    counts = df.groupby("flywire_id").size() / n_run / (T_MS / 1000.0)
    counts.index = counts.index.astype(np.int64).astype(str)
    return counts


def _run_once(g, groups, seed, device):
    st = stimulus.Stimulus([stimulus.Pulse(names, RATE_HZ, 0, T_MS, kind="poisson") for names in groups])
    return lif.LIF(g, seed=seed, device=device).run(st, T_MS, window_ms=50)


def run(graph=None, seeds=(0, 1, 2), save=True, device="auto"):
    t0 = time.time()
    g = graph or graph_mod.get(DATASET)
    sugar = populations.names(DATASET, "sugar_grn_right")
    bitter = populations.names(DATASET, "bitter_grn")
    orc = oracle_rates(paths.DATA / DATASET / "raw")
    ours, ours_sb, ids, device_used = [], [], [], None
    for s in seeds:
        r = _run_once(g, [sugar], s, device); r.extra["gate"] = "gate2_sugar"
        rb = _run_once(g, [sugar, bitter], s, device); rb.extra["gate"] = "gate2_sugar_bitter"
        ours.append(r.rates_hz().mean(axis=0)); ours_sb.append(rb.hz([MN9])); device_used = rb.extra["device"]
        if save:
            ids += [r.save().name, rb.save().name]
    hz = np.mean(ours, axis=0)
    ours_s = pd.Series(hz, index=g.names)
    top = orc.sort_values(ascending=False).index[:200]
    top = [t for t in top if t in ours_s.index]
    rho = float(spearmanr(ours_s[top].values, orc[top].values).correlation)
    mn9_o, mn9_e = float(ours_s[MN9]), float(orc.get(MN9, 0.0))
    mn9_sb = float(np.mean(ours_sb))
    out = {"mn9_ours_hz": mn9_o, "mn9_oracle_hz": mn9_e, "ratio": mn9_o / mn9_e if mn9_e else float("inf"),
           "spearman_top200": rho, "active_ours": int((ours_s > 1).sum()), "active_oracle": int((orc > 1).sum()),
           "mn9_sugar_bitter_hz": mn9_sb, "suppression": mn9_sb / mn9_o if mn9_o else float("inf"),
           "run_ids": ids, "device": device_used, "seconds": time.time() - t0}
    # informational only (not part of the criteria): comparison with a locally reproduced Brian2 run (see gate2
    # fix round 1 in the journal) — if the file exists we do not rerun the gate, just read the saved parquet.
    local = paths.RESULTS / "oracle_brian2" / "sugarR_ours.parquet"
    if local.exists():
        orc_local = oracle_rates(paths.RESULTS / "oracle_brian2", filename="sugarR_ours.parquet")
        mn9_local = float(orc_local.get(MN9, 0.0))
        out["mn9_local_brian2_hz"] = mn9_local
        out["ratio_vs_local_brian2"] = mn9_o / mn9_local if mn9_local else float("inf")
    act_ratio = out["active_ours"] / max(1, out["active_oracle"])
    out["passed"] = bool(0.6 <= out["ratio"] <= 1.4 and rho >= 0.7 and 0.7 <= act_ratio <= 1.4 and out["suppression"] <= 0.5)
    return out


if __name__ == "__main__":
    print(run())
