# Why sugar → MN9 in BANC is 4× weaker (FAFB 76.7 Hz, BANC 17.8, MaleCNS 68.9)

Atlas runs: results/20260915-013801-00 (fafb_783), -013810-00 (banc_888), -013831-00 (malecns_09),
LIF Shiu 2024, sugar_grn_right 150 Hz Poisson 250–750 ms, measurement window 300–750 ms.
Script: `lab/analysis/banc_sugar_mn9.py` (steps 1–4: `python lab/analysis/banc_sugar_mn9.py`, with control
runs: `--runs`). All numbers below are from its output.

## Summary

1. **This is not an artefact of our pipeline.** The min_count=5 threshold, the sign rule, and the group
   definitions were each checked separately — none of them explains the difference (details in §4–5, control
   runs in §6).
2. **Wiring is identical across all three scans** — same cell types, same sides: sugar GRN → layer 1 (CB0393, CB0616, CB0192, CB0366, AN_GNG_21/30) →
   excitatory premotor-nucleus loop (CB0553, DNge059, DNge080, CB0824, CB0051) → outputs onto MN9 (CB0553, DNge062, CB0493 — all
   cholinergic, contralateral) with inhibition from CB0465, DNge051, CB0903, CB0862, CB0806 (GABA).
3. **The difference is in the number of synapses on the same cells.** In BANC, cells on this path carry on average 0.39× the synapses of FAFB
   (MaleCNS — 1.11×). Within the excitatory loop: FAFB 6327 E→E synapses, BANC 1851 (0.29), MaleCNS 6585; E→MN9: 4480 / 1264 (0.28) / 1675.
4. The Shiu model weighs each synapse's contribution in absolute units (0.275 mV × synapse count), so the loop does not "ignite" in BANC:
   CB0553 22 Hz instead of 69, DNge062 13 instead of 33, CB0493 2 instead of 44, DNge059/DNge080/CB0824 — 0 Hz (36–93 Hz in FAFB).
   MN9 left receives Σ(w·Hz) = +6651 vs +64184 in FAFB (≈1/10), MN9 right gets nothing (0 Hz): its left-side drivers are silent.
5. Direct test: FAFB with weight 0.35·w_syn (BANC density) → MN9 = 0 Hz, 42 active cells instead of 330. The reverse — BANC with
   weight ×2…×4 — does not restore MN9 (10 / 14 / 2 Hz), because the synapse deficit in BANC is uneven: MN9's excitatory drivers
   (DNge062 0.27, DNge080 0.21) lost more than the inhibitors (CB0903 0.54, CB0248 0.79), and global scale-up raises inhibition sooner.

## 1. Wiring path (step 1)

Layer 1 = cells with ≥5 positive synapses from the sugar_grn_right group; B = direct positive inputs to MN9 (≥5); a length-3 path = GRN → A → B → MN9.

| | FAFB | BANC | MaleCNS |
| --- | --- | --- | --- |
| sugar_grn_right, cells | 30 | 49 | 27 |
| layer 1: cells / synapses from GRN | 107 / 3696 | 119 / 5388 | 170 / 9794 |
| direct + inputs to MN9 (≥5), cells | 65 | 44 | 45 |
| 2-step intermediaries (GRN→A→MN9) | 2 (CB0573 26/8, DNge036 5/5) | 2 (CB0393 234/10, CB0919 34/5) | 1 (CB0573 55/5) |
| 3-step A→B edges / Σ min over paths | 101 / 901 | 56 / 553 | 192 / 1658 |

There are no 2-step paths above noise in any scan (5–15 synapses — noise); the signal runs through ≥3 steps. Summary of 3-step paths by type B
(cells; paths; Σ min(GRN→A, A→B, B→MN9); B→MN9 synapses):

| type B | FAFB | BANC | MaleCNS |
| --- | --- | --- | --- |
| CB0553 | 2; 7; 98; **1316** | 2; 7; 61; **385** | 2; 16; 234; 375 |
| DNge062 | 1; 3; 22; **907** | 1; 2; 64; **201** | 2; 11; 108; 559 |
| CB0493 | 1; 3; 31; **431** | 1; 2; 47; **105** | 2; 2; 41; 440 |
| DNge080 | 2; 6; 76; 441 | 1; 1; 17; 60 | 2; 13; 201; 219 |
| CB0251 | 2; 15; 107; 72 | 0 | 2; 24; 199; 40 |
| DNge059 | 2; 5; 46; 103 | 0 (none on the right) | 2; 7; 92; 56 |
| CB0393 | 0 | 2; 19; 165; 16 | 1; 6; 37; 7 |
| CB0573 | 1; 10; 67; 8 | 0 | 1; 18; 90; 5 |

The three main drivers of MN9 (CB0553, DNge062, CB0493) are present in all scans, but in BANC they give MN9 3–4.5× fewer synapses
(1316→385, 907→201, 431→105). The raw numbers (before the threshold of 5) are the same: DNge062→MN9 900/841 in FAFB, 190/230 in BANC, 464/34 in MaleCNS;
CB0553→MN9 691/619, 176/191, 348/20. So the threshold is not the cause here — edges of 100–200 synapses clear it comfortably.

## 2. Activity path (step 2)

Active cells (>5 Hz in the sugar window) by BFS layer from sugar_grn_right (positive-sign graph):

| | FAFB | BANC | MaleCNS |
| --- | --- | --- | --- |
| MN9 left / right, Hz | 88.9 / 64.4 | **35.6 / 0.0** | 124.4 / 13.3 |
| active, total | 330 | 142 | 2020 |
| layer 1 (of layer size) | 35 / 101 | 37 / 84 | 65 / 147 |
| layer 2 | 152 / 2452 | **38 / 1446** | 381 / 2369 |
| layer 3 | 89 | 18 | 1106 |

**Layer 1 in BANC is no worse than FAFB** (37 active, CB0393 122 Hz, CB0366 113, AN_GNG_21 100, CB0248 171 — the same types are 50–130 Hz in FAFB).
**The break is at layer 2**, in the excitatory premotor-nucleus loop that in FAFB drives the MN9 outputs:

| cell (right side, contralateral to left MN9) | FAFB Hz | BANC Hz | MaleCNS Hz | input synapses + (FAFB / BANC / MaleCNS) |
| --- | --- | --- | --- | --- |
| CB0553 | 68.9 | 22.2 | 66.7 | 2003 / 895 / 3276 |
| DNge062 | 33.3 | 13.3 | 48.9 | 1205 / 265 / 1428 |
| CB0493 | 44.4 | 2.2 | 37.8 | 1817 / 500 / 2121 |
| DNge059 | 93.3 | no cell (left 0.0) | 57.8 | 3006 / — (790 on the left) / 3454 |
| DNge080 | 51.1 | 0.0 | 62.2 | 1882 / 276 / 1738 |
| CB0824 | 35.6 | 0.0 | 33.3 | 626 / 254 / 857 |
| CB0051 | 22.2 | 55.6 | 24.4 | 646 / 363 / 1155 |

The direct layer-1 input to CB0553 is small in all three scans (FAFB 71 synapses, BANC 53, MaleCNS 250); in FAFB CB0553 is driven up by feedback:
CB0824→CB0553 324 synapses (while CB0553→CB0824 only 118), DNge059→CB0553 110 while DNge059 itself runs at 93 Hz (held up by DNge080 275 + CB0553 102),
CB0051→CB0553 232. In BANC: CB0824→CB0553 289 is present, but CB0553→CB0824 is only 18–36 vs 94–118 in FAFB (CB0824 is silent); DNge059 is
entirely absent on the right (a single left-side cell, status LR_TYPE_CONFLICT/FAFB_TYPE_CONFLICT); CB0051→CB0553 73. The loop does not close,
CB0553 stays on its direct input alone — 22 Hz. In addition, in BANC CB0553(r) is inhibited by two cells that in this form do not exist in FAFB:
`auto:DNge062`(r) (GABA, 127 Hz, an auto-assigned label under the cholinergic type DNge062 — suspect, status NECK_CHECKED_NOT_AN_DN) w=−42, and
DNge146(r) w=−176 (DNge146 is silent in FAFB).

Contribution to left MN9: Σ(w·Hz) FAFB +119811 / −55627 = +64184; BANC +6651 / 0 = +6651 (CB0553 3911, DNge062 2533, CB0493 207);
MaleCNS +95969 / −32836 = +63133. Right MN9 in BANC: +1607 / −2953 — its drivers (CB0553/DNge062/CB0493 on the left, 191/230/98
synapses in the raw list) sit at 0 Hz because the stimulus is right-only and the left half of the loop never starts.

## 3. Annotation check (step 3)

- **Transmitter / sign = 0 is not the cause.** Among MN9 inputs (in pairs ≥5), the share of synapses from unsigned cells: FAFB 0.0 %, BANC 0.0 %, MaleCNS 0.8 %.
  Among layer-1 inputs: 0.0 / 0.3 / 6.1 %. 18.3 % of BANC cells have an empty/unclear transmitter (FAFB 3.4 %, MaleCNS 9.0 %), but these are glia,
  visual and small fragments — not path cells. All 19 path types have matching transmitters between BANC and FAFB (ACh/GABA), except CB0824:
  in BANC 4 cells with ACh/GABA/glutamate (in FAFB 2, both ACh) — one of the "extra" left CB0824 is glutamatergic, sign −1, but it is silent.
- **Types exist and are labeled.** 98.3 % of BANC CB/DN/AN cells carry fafb_cell_type; every FAFB-path type is found. Gaps: DNge059 — one cell
  (left) instead of two; CB0862 and CB0499 — one each; CB0493 on the right has input_connections=0 in meta (500 inputs in the edgelist — meta
  and edgelist disagree); extra auto:CB0393 (glia) and auto:CB0493 (TOO_SMALL) are not included in the groups (absent from mn9/sugar).
- **The min_count=5 threshold is secondary.** Synapses in pairs ≥5: MN9 inputs — FAFB 97 %, BANC 92 %, MaleCNS 88 %; layer-1 inputs — 86 / 74 / 77 %;
  across the whole edgelist — 70 / 56 / 32 %. BANC loses slightly more at the threshold than FAFB (small pairs), but MaleCNS loses even more and
  still responds normally.
- **The sugar_grn_right group.** 49 BANC cells vs 30 FAFB and 27 MaleCNS; output per cell is similar (132 / 136 / 390 synapses ≥5), and total
  input to layer 1 in BANC is actually larger than FAFB (5388 vs 3696). GRN transmitter: BANC 37 ACh + 11 glutamate (sign −1) + 1 octopamine; FAFB 19 ACh + 7 serotonin + 4 glutamate;
  MaleCNS 20 ACh + 5 unclear (sign 0) + 2 glutamate. Negative contribution to layer 1: −134 (BANC), −52, −80 — small against +5388.

## 4. Input and synapse density (step 4)

MN9 per stimulated GRN: FAFB 2.56 Hz, BANC 0.36, MaleCNS 2.55 — normalizing for group size makes the gap 7-fold, it does not remove it.

Raw input synapses per cell (mean per type, edgelist before threshold):

| type | FAFB | BANC | MaleCNS | BANC/FAFB |
| --- | --- | --- | --- | --- |
| MN9 | 8021 | 2816 | 3250 (left ≈ 5900, right barely assembled) | 0.35 |
| DNge062 | 2560 | 679 | 2684 | 0.27 |
| DNge080 | 4254 | 879 | 4032 | 0.21 |
| DNge059 | 7520 | 2330 | 8697 | 0.31 |
| CB0553 | 5668 | 2520 | 6216 | 0.44 |
| CB0493 | 5106 | 1744 | 4764 | 0.34 |
| CB0824 | 2166 | 778 | 2409 | 0.36 |
| CB0465 (GABA) | 2122 | 854 | 3819 | 0.40 |
| CB0903 (GABA) | 3686 | 2004 | 4482 | 0.54 |
| CB0248 (GABA, layer 1) | 1799 | 1428 | 2528 | 0.79 |
| CB0393 (layer 1) | 4588 | 2786 | 4732 | 0.61 |
| median over 19 types | | | | **0.39** (MaleCNS 1.11) |

The same holds brain-wide: median input synapses for central_brain+descending cells is FAFB 422, BANC 207, MaleCNS 714; the whole BANC edgelist —
42.3 million synapses vs 68.9 million in FAFB (even though BANC also covers the ventral nerve cord). That is, in BANC v888 (edgelist_simple_v3, Lee-lab)
roughly half as many synapses are detected/assigned to subtracted cells overall, and in the subesophageal ganglion (the path cells) — 2.5–5× fewer
than in FAFB v783; layer 1 (CB0393 0.61, CB0248 0.79) fared better than the premotor loop and descending neurons (0.21–0.35).

The whole excitatory loop (after threshold, as in the simulator; E = CB0553, DNge059, DNge080, CB0824, CB0051, DNge062, CB0493, CB0855):

| | E→E | layer1→E | I→E | E→MN9 | I→MN9 | E/I onto MN9 |
| --- | --- | --- | --- | --- | --- | --- |
| FAFB | 6327 | 509 | 6374 | 4480 | 5270 | 0.85 |
| BANC | **1851** | 331 | 1457 | **1264** | 1754 | 0.72 |
| MaleCNS | 6585 | 1487 | 7168 | 1675 | 2105 | 0.80 |

## 5. Where exactly the signal is lost in BANC

| layer | what | cause |
| --- | --- | --- |
| GRN → layer 1 | no loss: 37 active, 50–170 Hz | — |
| layer 1 → loop (CB0553/DNge062/CB0493) | direct input is weak in every scan, the loop fails to ignite in BANC: 22 / 13 / 2 Hz | synapses inside the loop are 0.29× FAFB (reconstruction/synapse detection), DNge059 missing on the right (annotation/reconstruction), CB0553→CB0824 18 instead of 118 |
| loop → left MN9 | 1264 synapses instead of 4480 at roughly half the presynaptic firing rate → drive 1/10 | same cells, 3–4.5× fewer synapses per edge |
| loop → right MN9 | 0 Hz | left-side drivers are at 0 Hz (stimulus is one-sided, the left loop never starts); in FAFB the left loop is picked up via crossing connections CB0553(r)→CB0493(l) 218, DNge059(r)→…, which are 70 and 0 in BANC |

At no step is "no cells", "no label", or "no transmitter" the sole cause; the only clean annotation gap is DNge059 on the right.

## 6. Control runs (step 5; runner.run_experiment, YAML copies in a temp folder, save=True)

| variant | run | MN9 sugar, Hz (left/right) | active cells |
| --- | --- | --- | --- |
| atlas (49 GRNs) | 20260915-013810-00 | 17.8 (35.6 / 0) | 142 |
| sugar_grn_right → 30 random of 49 (seed 0) | 20260915-032116-00 (duplicate 031533-00) | **7.8** (15.6 / 0) | 109 |
| sugar_grn_all (540 cells, incl. legs/wings) | 20260915-032126-00 | 18.9 (37.8 / 0) | 909 |
| BANC, w_syn ×2 | 20260915-032135-00 | 10.0 (17.8 / 2.2) | 5165 |
| BANC, w_syn ×2.85 (=1/0.35) | 20260915-032144-00 | 14.4 (17.8 / 11.1) | 8617 |
| BANC, w_syn ×4 | 20260915-032154-00 | 2.2 (0 / 4.4) | 21043 |
| FAFB, w_syn ×0.35 (BANC density) | 20260915-032207-00 | **0.0** | 42 |

Conclusions: (1) the group is not the cause: fewer GRNs → even weaker, more GRNs (×11) → the same 18.9 Hz; (2) FAFB with synapse counts cut down
to BANC density fails to drive MN9 at all — the Shiu model depends critically on absolute synapse count; (3) simple inverse rescaling of BANC does
not fix it: at ×2.85, CB0553 is 24 Hz, DNge062 13, and the inhibitor CB0465 47 Hz (Σ onto left MN9 +14518 / −16667) — scale-up raises inhibition
sooner, and overall brain activity too (8617 active cells vs 330 in FAFB), because the synapse deficit in BANC is uneven.

## 7. Verdict (probabilities are subjective estimates based on this data)

- **(b) incomplete reconstruction / synapse detection in BANC v888 — 0.70.** Same cells, same types, same signs, but 0.2–0.45× FAFB synapses
  across the whole path and 0.5× brain-wide; MaleCNS, loaded with the same rules, gives 1.1× FAFB and responds like FAFB.
- **(a) BANC annotation error/gap — 0.15.** DNge059 is missing on the right (type conflict), `auto:DNge062` is a GABAergic cell under a cholinergic
  type, CB0824 has 4 cells with three transmitters. This finishes off the loop, but even without it the synapse count is a third of FAFB.
- **(d) an artefact of our pipeline — 0.10.** Not the threshold (paths of 100–900 synapses), not the sign (0 % unsigned synapses at MN9 inputs),
  not the group (controls). What remains is a "model artefact": the absolute weight of 0.275 mV/synapse, fitted by Shiu to FAFB density, without
  normalization per scan. This is not a code bug but a limitation of transferring the model between scans — and, strictly speaking, this is the
  mechanism through which (b) manifests.
- **(c) a real wiring difference — 0.05.** Topology and sides match; only the weights differ — uniformly across all cells at once, which is
  atypical for biological variation.

## 8. Next experiment (one)

Normalize BANC not globally but **per cell input**: instead of W = sign·synapse count, use W = sign·synapse count × (median FAFB input
for the same fafb_cell_type / this cell's input in BANC) — i.e. restore each path cell's FAFB density
(Lee-lab's edgelist has a `norm` column = the postsynaptic cell's input share, so one can take `norm × FAFB-type input`). Rerun the same
experiment on fly_taste_banc_888; expectation under (b): left MN9 ≥ 50 Hz, CB0553/DNge062/CB0493 30–70 Hz, ~300 active cells, without a
brain-wide blow-up (unlike the global ×2.85). If this still fails to raise MN9, the cause is annotation (DNge059/auto:DNge062), and the next
step is manual review of these three types in BANC Codex. Requires a new graph-assembly method (a dataset parameter), i.e. changes to
brainlab/store — outside the scope of this analysis.
