"""Анализ: почему в BANC сахар → MN9 в 4 раза слабее, чем в FAFB и MaleCNS (атлас 2026-09-15).

Запуск из корня проекта:
    python lab/analysis/banc_sugar_mn9.py            # шаги 1–4 (проводка, активность, разметка, вход)
    python lab/analysis/banc_sugar_mn9.py --runs     # + шаг 5: контрольные прогоны BANC на GPU (сохраняются в results/)

Ничего в brainlab/ не меняет; читает граф (brainlab.store.graph), базу подписей, прогоны атласа и
сырые edgelist-файлы Lee-lab (data/<набор>/raw), чтобы посмотреть синапсы ДО порога min_count=5.
Отчёт: docs/2026-09-15-анализ-banc-сахар-mn9.md.
"""
import argparse
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as ft
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from brainlab import paths                      # noqa: E402
from brainlab.lab import populations, runner    # noqa: E402
from brainlab.sim import result                 # noqa: E402
from brainlab.store import db, graph            # noqa: E402

DS = ["fafb_783", "banc_888", "malecns_09"]
RUNS = {"fafb_783": "20260915-013801-00", "banc_888": "20260915-013810-00", "malecns_09": "20260915-013831-00"}
EDGES = {"fafb_783": "fafb_783_simple_edgelist.feather", "banc_888": "banc_888_edgelist_simple_v3.feather",
         "malecns_09": "malecns_09_simple_edgelist.feather"}
SIGNED = {"acetylcholine", "gaba", "glutamate", "histamine", "dopamine", "serotonin", "octopamine", "tyramine"}
SUGAR_WIN = (6, 15)        # окна 50 мс: 300–750 мс (как measure.sugar в YAML опыта)
ACTIVE_HZ = 5.0
# типы пути сахар → MN9 (по FAFB): слой 2 = прямые входы MN9, слой 1/петля = их возбудители
PATH_TYPES = ["MN9", "DNge062", "CB0553", "CB0493", "CB0465", "DNge051", "CB0903", "CB0862", "CB0806",
              "CB0824", "DNge059", "DNge080", "CB0051", "CB0855", "CB0393", "CB0616", "CB0192", "CB0499", "CB0248"]

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)


class Scan:
    """Граф + подписи + прогон атласа одного набора, с удобными подписями клеток."""

    def __init__(self, ds, conn):
        self.ds = ds
        self.g = graph.get(ds)
        self.ns = {x["name"]: x for x in db.neurons(conn, ds)}
        self.r = result.load(paths.RESULTS / RUNS[ds])
        assert self.r.names == self.g.names
        hz = self.r.rates_hz()
        self.sugar_hz = hz[SUGAR_WIN[0]:SUGAR_WIN[1]].mean(0)
        self.sugar = populations.names(ds, "sugar_grn_right")
        self.mn9 = populations.names(ds, "mn9")
        self.si = self.g.idx(self.sugar)
        self.mi = self.g.idx(self.mn9)
        self.W = self.g.W_chem
        self.Wp = self.W.maximum(0).tocsr()
        x = np.zeros(self.g.n); x[self.si] = 1
        self.from_grn_pos = self.Wp @ x          # + синапсы от группы sugar_grn_right на каждую клетку
        self.from_grn_signed = self.W @ x
        y = np.zeros(self.g.n); y[self.mi] = 1
        self.to_mn9_pos = self.Wp.T @ y           # + синапсы от клетки на группу MN9
        self.layer = self._layers()

    def ftype(self, i):
        x = self.ns[self.g.names[i]]
        if x["cell_type"] == "MN9":          # у MaleCNS fafb_cell_type MN9 = CB0701, а нужен сам MN9
            return "MN9"
        return x["extra"].get("fafb_cell_type") or x["cell_type"] or "-"

    def label(self, i):
        return "%s(%s)" % (self.ftype(i), (self.ns[self.g.names[i]]["side"] or "?")[:1])

    def _layers(self):
        """Слой по BFS в положительном графе от sugar_grn_right (0 — сами ГРН)."""
        layer = np.full(self.g.n, 99); layer[self.si] = 0; front = self.si
        for L in range(1, 5):
            x = np.zeros(self.g.n); x[front] = 1
            new = np.where((self.Wp @ x > 0) & (layer == 99))[0]
            layer[new] = L; front = new
        return layer


# ---------------------------------------------------------------- шаг 1: путь по проводке
def step1_paths(scans):
    print("\n### Шаг 1. Пути по проводке (положительные знаки), sugar_grn_right → … → mn9")
    rows = []
    for s in scans.values():
        A = np.where(s.from_grn_pos >= 5)[0]; B = np.where(s.to_mn9_pos >= 5)[0]
        two = [i for i in A if s.to_mn9_pos[i] > 0]
        print("%s: слой 1 (≥5 синапсов от ГРН): %d клеток, %d синапсов; прямые входы MN9 (≥5, +): %d клеток; "
              "2-шаговых промежуточных: %d (%s)" % (
                  s.ds, len(A), int(s.from_grn_pos[A].sum()), len(B), len(two),
                  ", ".join("%s %d/%d" % (s.label(i), s.from_grn_pos[i], s.to_mn9_pos[i]) for i in two)))
        sub = s.Wp[B][:, A].tocoo()      # рёбра A→B, 3-шаговые пути ГРН→A→B→MN9
        agg = {}
        for bi, ai, w in zip(sub.row, sub.col, sub.data):
            b, a = B[bi], A[ai]
            d = agg.setdefault(s.ftype(b), {"n_paths": 0, "sum_min": 0.0, "cells": set()})
            d["n_paths"] += 1; d["sum_min"] += min(s.from_grn_pos[a], w, s.to_mn9_pos[b]); d["cells"].add(b)
        for t, d in agg.items():
            rows.append({"scan": s.ds, "type_B": t, "cells": len(d["cells"]), "paths": d["n_paths"], "sum_min": int(d["sum_min"]),
                         "to_mn9": int(sum(s.to_mn9_pos[i] for i in d["cells"]))})
        print("   3-шаговых рёбер A→B: %d, сумма min по путям: %d" % (sub.nnz, sum(d["sum_min"] for d in agg.values())))
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index="type_B", columns="scan", values=["cells", "paths", "sum_min", "to_mn9"], aggfunc="first").fillna(0).astype(int)
    piv["total"] = piv["sum_min"].sum(axis=1)
    print(piv.sort_values("total", ascending=False).head(16).to_string())
    return piv


# ---------------------------------------------------------------- шаг 2: путь по активности
def step2_activity(scans):
    print("\n### Шаг 2. Активность (> %.0f Гц в окне сахара 300–750 мс) по слоям BFS" % ACTIVE_HZ)
    for s in scans.values():
        act = np.where(s.sugar_hz > ACTIVE_HZ)[0]
        by = {int(L): int((s.layer[act] == L).sum()) for L in sorted(set(s.layer[act]))}
        size = {int(L): int((s.layer == L).sum()) for L in (1, 2, 3)}
        print("%s: MN9 = %s Гц; активных %d, по слоям %s (размер слоёв 1/2/3: %s)" % (
            s.ds, np.round(s.sugar_hz[s.mi], 1).tolist(), len(act), by, size))
        for k, i9 in enumerate(s.mi):
            row = s.W[i9].toarray().ravel(); pre = np.where(row != 0)[0]
            drive = row * s.sugar_hz
            top = sorted(pre, key=lambda j: -abs(drive[j]))[:6]
            print("   MN9 %s: входов %d, активных %d, Σ(w·Гц) = %+.0f (+%.0f / %.0f); главные: %s" % (
                s.ns[s.mn9[k]]["side"], len(pre), int((s.sugar_hz[pre] > ACTIVE_HZ).sum()), drive.sum(),
                drive[drive > 0].sum(), drive[drive < 0].sum(),
                ", ".join("%s w=%d %.0fГц" % (s.label(j), row[j], s.sugar_hz[j]) for j in top)))
    print("\nКлючевые клетки пути (тип(сторона): Гц | вх.синапсов +/− | Σ(w·Гц) +/− | главные входы):")
    for t in ["CB0553", "DNge062", "CB0493", "DNge059", "CB0824", "DNge080", "CB0051"]:
        for s in scans.values():
            for i in [i for i in range(s.g.n) if s.ftype(i) == t]:
                row = s.W[i].toarray().ravel(); pre = np.where(row != 0)[0]; d = row * s.sugar_hz
                top = sorted(pre, key=lambda j: -abs(d[j]))[:4]
                print("   %-11s %-16s %6.1f | %5d/%5d | %+7.0f/%+7.0f | %s" % (
                    s.ds, s.label(i), s.sugar_hz[i], row[row > 0].sum(), -row[row < 0].sum(), d[d > 0].sum(), d[d < 0].sum(),
                    ", ".join("%s w=%d %.0fГц" % (s.label(j), row[j], s.sugar_hz[j]) for j in top)))


# ---------------------------------------------------------------- сырые edgelist: синапсы до порога
def raw_edges_to(ds, targets):
    """Все сырые рёбра (pre, post, count) на клетки targets (до порога min_count)."""
    mm = pa.memory_map(str(paths.DATA / ds / "raw" / EDGES[ds]), "r"); r = pa.ipc.open_file(mm)
    vs = pa.array(list(targets)); parts = []
    for i in range(r.num_record_batches):
        b = r.get_batch(i); post = b.column("post").cast(pa.string())
        m = pc.is_in(post, value_set=vs)
        if pc.any(m).as_py():
            parts.append(pa.table({"pre": b.column("pre").cast(pa.string()), "post": post, "count": b.column("count")}).filter(m).to_pandas())
    mm.close()
    return pd.concat(parts) if parts else pd.DataFrame(columns=["pre", "post", "count"])


def raw_totals(ds, ids):
    """Сырые суммы синапсов (вход/выход, всего и в парах ≥5) на клетку по всему edgelist; ids — Index строк."""
    mm = pa.memory_map(str(paths.DATA / ds / "raw" / EDGES[ds]), "r"); r = pa.ipc.open_file(mm)
    n = len(ids); acc = {k: np.zeros(n) for k in ("in_raw", "in_ge5", "out_raw", "out_ge5")}; tot = 0; pairs = 0; syn5 = 0
    for i in range(r.num_record_batches):
        b = r.get_batch(i)
        pre = ids.get_indexer(b.column("pre").to_numpy(zero_copy_only=False)); post = ids.get_indexer(b.column("post").to_numpy(zero_copy_only=False))
        c = b.column("count").to_numpy().astype(np.int64); c5 = np.where(c >= 5, c, 0)
        tot += c.sum(); pairs += len(c); syn5 += c5.sum()
        ok = (pre >= 0) & (post >= 0)
        acc["in_raw"] += np.bincount(post[ok], weights=c[ok], minlength=n); acc["in_ge5"] += np.bincount(post[ok], weights=c5[ok], minlength=n)
        acc["out_raw"] += np.bincount(pre[ok], weights=c[ok], minlength=n); acc["out_ge5"] += np.bincount(pre[ok], weights=c5[ok], minlength=n)
    mm.close()
    return pd.DataFrame(acc, index=ids), {"syn_total": int(tot), "pairs": int(pairs), "syn_ge5": int(syn5)}


# ---------------------------------------------------------------- шаг 3: разметка, медиатор, порог
def step3_labels(scans):
    print("\n### Шаг 3. Разметка: медиатор (знак 0), порог min_count, наличие типов FAFB-пути")
    for s in scans.values():
        l1 = [s.g.names[i] for i in np.where(s.from_grn_pos >= 5)[0]]
        E = raw_edges_to(s.ds, set(s.mn9) | set(l1))
        E["tr"] = E.pre.map(lambda p: s.ns[p]["transmitter"] if p in s.ns else "<нет в meta>")
        E["signed"] = E.tr.isin(SIGNED)
        for name, sub in [("входы MN9", E[E.post.isin(s.mn9)]), ("входы слоя 1 (%d кл.)" % len(l1), E[E.post.isin(l1)])]:
            tot = sub["count"].sum(); s5 = sub[sub["count"] >= 5]; k5 = s5["count"].sum(); z = s5.loc[~s5.signed, "count"].sum()
            print("   %-11s %-24s сырых %7d; в парах ≥5: %7d (%.0f%%); из них со знаком 0: %6d (%.1f%%)" % (
                s.ds, name, tot, k5, 100 * k5 / max(tot, 1), z, 100 * z / max(k5, 1)))
        tr = pd.Series([x["transmitter"] for x in s.ns.values()])
        trs = pd.Series([s.ns[n]["transmitter"] for n in s.sugar]).value_counts().to_dict()
        print("   %-11s медиатор пустой/unclear у %.1f%% всех клеток; sugar_grn_right: %s" % (s.ds, 100 * (~tr.isin(SIGNED)).mean(), trs))
        if s.ds != "fafb_783":
            cb = [x for x in s.ns.values() if x["cell_class"] in ("central_brain_intrinsic", "descending", "ascending")]
            has = sum(1 for x in cb if x["extra"].get("fafb_cell_type"))
            print("   %-11s CB/DN/AN клеток %d, с fafb_cell_type %.1f%%" % (s.ds, len(cb), 100 * has / len(cb)))
    # присутствие типов пути и стороны
    print("   Типы пути: число клеток (по fafb_cell_type/cell_type) и стороны")
    for t in PATH_TYPES:
        line = []
        for s in scans.values():
            cells = [i for i in range(s.g.n) if s.ftype(i) == t]
            sides = "".join(sorted((s.ns[s.g.names[i]]["side"] or "?")[:1] for i in cells))
            line.append("%s %d [%s]" % (s.ds[:4], len(cells), sides))
        print("      %-8s %s" % (t, " | ".join(line)))


# ---------------------------------------------------------------- шаг 4: вход и плотность синапсов
def step4_input(scans, conn):
    print("\n### Шаг 4. Вход: размер группы, выход ГРН, плотность синапсов на одних и тех же типах")
    for s in scans.values():
        out = np.abs(s.W[:, s.si]).sum()
        print("   %-11s sugar_grn_right %2d кл., Σ|синапсов наружу| (≥5) = %5d, на 1 ГРН %.0f; на слой 1: +%d / %d; "
              "MN9 = %.1f Гц → %.2f Гц на 1 ГРН" % (
                  s.ds, len(s.sugar), out, out / len(s.sugar), s.from_grn_pos.sum(), s.from_grn_signed[s.from_grn_signed < 0].sum(),
                  s.sugar_hz[s.mi].mean(), s.sugar_hz[s.mi].mean() / len(s.sugar)))
    res = {}
    for s in scans.values():
        t0 = time.time()
        ids = pd.Index(s.g.names)
        tot, glob = raw_totals(s.ds, ids)
        cls = np.array(s.g.cell_class); cb = tot[np.isin(cls, ["central_brain_intrinsic", "descending"])]
        print("   %-11s edgelist: %d синапсов в %d парах, в парах ≥5 — %.0f%%; клетки CB+DN (n=%d): медиана вх. синапсов %.0f (≥5: %.0f) [%.0f с]" % (
            s.ds, glob["syn_total"], glob["pairs"], 100 * glob["syn_ge5"] / glob["syn_total"], len(cb), cb.in_raw.median(), cb.in_ge5.median(), time.time() - t0))
        ftype = pd.Series([s.ftype(i) for i in range(s.g.n)], index=ids)
        sel = tot[ftype.isin(PATH_TYPES)].assign(ft=ftype[ftype.isin(PATH_TYPES)])
        res[s.ds] = sel.groupby("ft").agg(n=("in_raw", "size"), in_raw=("in_raw", "mean"), in_ge5=("in_ge5", "mean"))
    T = pd.concat(res, axis=1); T.columns = ["%s_%s" % (a[:4], b) for a, b in T.columns]
    T["banc/fafb"] = (T["banc_in_raw"] / T["fafb_in_raw"]).round(2); T["male/fafb"] = (T["male_in_raw"] / T["fafb_in_raw"]).round(2)
    print("   Сырые входные синапсы на клетку (среднее по типу):")
    T[[c for c in T.columns if "/" not in c]] = T[[c for c in T.columns if "/" not in c]].round(0)
    print(T.to_string())
    print("   медиана отношения banc/fafb по типам: %.2f, male/fafb: %.2f" % (T["banc/fafb"].median(), T["male/fafb"].median()))
    return T


# ---------------------------------------------------------------- шаг 4б: возбуждающая петля премоторного ядра
LOOP_E = ["CB0553", "DNge059", "DNge080", "CB0824", "CB0051", "DNge062", "CB0493", "CB0855"]
LOOP_I = ["CB0465", "DNge051", "CB0903", "CB0862", "CB0806", "DNge031", "DNge146", "CB0219", "CB3892b"]


def step4b_loop(scans):
    """Синапсы внутри возбуждающей петли (типы, которые в FAFB держат CB0553/DNge062/CB0493 на 30–70 Гц)
    и на MN9 — та же проводка в трёх сканах, после порога min_count=5 (как в симуляторе)."""
    print("\n### Шаг 4б. Возбуждающая петля премоторного ядра (E = %s)" % ", ".join(LOOP_E))
    for s in scans.values():
        T = np.array([s.ftype(i) for i in range(s.g.n)])
        L = np.where(np.isin(T, LOOP_E))[0]; I = np.where(np.isin(T, LOOP_I))[0]
        Wn = -s.W.minimum(0)
        A = np.where(s.from_grn_pos >= 5)[0]
        ee, ie, ae = s.Wp[L][:, L].sum(), Wn[L][:, I].sum(), s.Wp[L][:, A].sum()
        em, im = s.Wp[s.mi][:, L].sum(), Wn[s.mi][:, I].sum()
        print("   %-11s E-клеток %d: E→E %5d | слой 1→E %4d | I→E %5d | E→MN9 %5d | I→MN9 %5d | E/I на MN9 %.2f" % (
            s.ds, len(L), ee, ae, ie, em, im, em / max(im, 1)))


# ---------------------------------------------------------------- шаг 5: контрольные прогоны
def step5_runs(which):
    print("\n### Шаг 5. Контрольные прогоны (runner.run_experiment, save=True → results/)")
    tmp = Path(tempfile.mkdtemp(prefix="brainlab_banc_controls_"))   # копии YAML во временной папке, не в lab/experiments
    print("   временные YAML опытов: %s" % tmp)
    base = yaml.safe_load((paths.LAB / "experiments" / "fly_taste_banc_888.yaml").read_text(encoding="utf-8"))
    fafb = yaml.safe_load((paths.LAB / "experiments" / "fly_taste_fafb_783.yaml").read_text(encoding="utf-8"))
    names = populations.names("banc_888", "sugar_grn_right")
    sub30 = sorted(random.Random(0).sample(names, 30))
    variants = {
        "banc_sugarR30_seed0": (base, {"sugar": sub30}),
        "banc_sugar_all": (base, {"sugar": "group:sugar_grn_all"}),
        "banc_wsyn_x2": (base, {"w_syn": 0.275 * 2}),
        "banc_wsyn_x2.85": (base, {"w_syn": 0.275 * 2.85}),
        "banc_wsyn_x4": (base, {"w_syn": 0.275 * 4}),
        "fafb_wsyn_x0.35": (fafb, {"w_syn": 0.275 * 0.35}),
    }
    out = []
    for name, (src, mod) in variants.items():
        if which and name not in which:
            continue
        spec = yaml.safe_load(yaml.safe_dump(src, allow_unicode=True))
        spec["name"] = "fly_taste_" + name
        if "sugar" in mod:
            for p in spec["stimulus"]["pulses"]:
                if p["names"] == "group:sugar_grn_right":
                    p["names"] = mod["sugar"]
        if "w_syn" in mod:
            spec["params"]["w_syn"] = float(mod["w_syn"])
        spec["notes"] = "Контроль анализа BANC (lab/analysis/banc_sugar_mn9.py, вариант %s)." % name
        f = tmp / (name + ".yaml"); f.write_text(yaml.safe_dump(spec, allow_unicode=True, sort_keys=False), encoding="utf-8")
        t0 = time.time(); r = runner.run_experiment(f, save=True); dt = time.time() - t0
        mn9 = populations.names(spec["dataset"], "mn9"); hz = r.rates_hz(); pos = {n: i for i, n in enumerate(r.names)}
        per = [round(float(hz[SUGAR_WIN[0]:SUGAR_WIN[1], pos[n]].mean()), 1) for n in mn9]
        n_act = int((hz[SUGAR_WIN[0]:SUGAR_WIN[1]].mean(0) > ACTIVE_HZ).sum())
        out.append({"variant": name, "run": r.id, "mn9_sugar_hz": round(r.hz(mn9, 300, 750), 1), "per_cell": per,
                    "mn9_sugar_bitter_hz": round(r.hz(mn9, 1050, 1500), 1), "active_cells": n_act, "flags": r.flags, "sec": round(dt, 1)})
        print("   %-20s прогон %s  MN9 сахар %.1f Гц %s, сахар+горечь %.1f Гц, активных клеток %d, flags %s, %.0f с" % (
            name, r.id, out[-1]["mn9_sugar_hz"], per, out[-1]["mn9_sugar_bitter_hz"], n_act,
            {k: v for k, v in r.flags.items() if k != "mean_rate"}, dt))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", action="store_true"); ap.add_argument("--only", nargs="*")
    ap.add_argument("--skip-raw", action="store_true", help="без чтения сырых edgelist (шаги 3–4 частично)")
    a = ap.parse_args()
    conn = db.connect()
    scans = {ds: Scan(ds, conn) for ds in DS}
    step1_paths(scans)
    step2_activity(scans)
    if not a.skip_raw:
        step3_labels(scans)
        step4_input(scans, conn)
    step4b_loop(scans)
    if a.runs:
        step5_runs(a.only)


if __name__ == "__main__":
    main()
