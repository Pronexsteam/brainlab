"""Ворота 2. Муха: рецепторы сахара → мотонейрон хоботка MN9; горечь глушит ответ.
Эталон — вывод модели Shiu и др. 2024 (Brian2) на тех же данных v630: sugarR.parquet из их
репозитория (21 сахарная клетка справа, Пуассон 150 Гц, 1 с, 30 повторов; MN9 93,3 ± 3,2 Гц).
Критерии: 0.6 ≤ MN9_ours/MN9_oracle ≤ 1.4; Спирмен по 200 самым активным клеткам эталона ≥ 0.7;
число активных (>1 Гц) клеток в 0.7–1.4 от эталона; сахар+горечь ≤ 0.5·сахар по MN9.
Пуассон у нас и у Brian2 — разные генераторы, поэтому сравнение по частотам и порядку, не по спайкам.
Оговорка: абсолютные частоты отгруженного parquet (GRN 197 Гц, MN9 93 Гц) авторским кодом на текущем
Brian2 2.10 не воспроизводятся (150 / 83 Гц), порядок клеток — воспроизводится (ρ 0,987 по топ-200).
Оговорка про локальный Brian2 (авторский код, 82,7 Гц): ratio ≈ 1,35 при границе критерия 1,4 —
наш прогон относительно локально воспроизведённого эталона ближе к границе, чем относительно
отгруженного parquet (ratio ≈ 1,2); критерий считается только по отгруженному parquet, локальный
Brian2 — информационная сверка (см. results/oracle_brian2/sugarR_ours.parquet, если он есть)."""
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
    """id → Гц: среднее число спайков на повтор из parquet эталона, делённое на t_run = 1 с."""
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
    # информационно (в критерии не входит): сравнение с локально воспроизведённым Brian2 (см. gate2
    # fix round 1 в журнале) — если файл есть, не перегоняем ворота, просто читаем сохранённый parquet.
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
