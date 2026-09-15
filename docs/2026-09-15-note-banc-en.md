# Sugar → MN9 in BANC v888 vs FAFB v783 and MaleCNS v0.9: a simulation observation

To the BANC team, cc Lee-lab. Draft, 2026-09-15; corrections welcome.

## Summary

With the Shiu et al. 2024 LIF model unchanged, labellar sugar GRN stimulation drives MN9 at 76.7 Hz in FAFB v783 and 68.9 Hz in MaleCNS v0.9, but 17.8 Hz in BANC v888. The sugar→MN9 path has the same cell types, sides and transmitters in all three; what differs is the synapse count on the same cells (BANC median 0.39× FAFB across 19 path types; MaleCNS 1.11×), and the deficit is present in the native `synapses_v3` table itself. Most likely it reflects synapse detection at the release threshold and/or incomplete proofreading of presynaptic fragments in the subesophageal region, plus a few labeling items.

## Setup

- Model: Shiu et al. 2024 (philshiu/Drosophila_brain_model). LIF, V_rest −52 mV, threshold −45 mV, τ_m 20 ms, τ_syn 5 ms, 0.275 mV/synapse, delay 1.8 ms, refractory 2.2 ms, dt 0.1 ms; no per-dataset tuning.
- Data: Lee-lab `compiled_data` edgelists (`fafb_783_simple_edgelist`, `banc_888_edgelist_simple_v3`, `malecns_09_simple_edgelist`) and `*_meta`; edges under 5 synapses dropped.
- Sign from predicted presynaptic transmitter: ACh +1; GABA/glutamate/histamine −1; monoamines +1; unclear 0.
- Protocol: right labellar sugar GRNs (30 / 49 / 27 cells in FAFB / BANC / MaleCNS), 150 Hz Poisson, 250–750 ms; MN9 rate over 300–750 ms.

## Observation 1: same path, 0.39× synapses

Identical wiring: sugar GRNs → layer 1 (CB0393, CB0616, CB0192, CB0366, AN_GNG_21/30) → excitatory premotor loop (CB0553, DNge059, DNge080, CB0824, CB0051) → cholinergic outputs onto MN9 (CB0553, DNge062, CB0493), inhibited by CB0465, DNge051, CB0903. Mean input synapses per cell:

| type | FAFB | BANC | MaleCNS | BANC/FAFB |
| --- | --- | --- | --- | --- |
| MN9 | 8021 | 2816 | 3250 | 0.35 |
| DNge062 | 2560 | 679 | 2684 | 0.27 |
| DNge080 | 4254 | 879 | 4032 | 0.21 |
| DNge059 | 7520 | 2330 | 8697 | 0.31 |
| CB0553 | 5668 | 2520 | 6216 | 0.44 |
| CB0493 | 5106 | 1744 | 4764 | 0.34 |
| CB0824 | 2166 | 778 | 2409 | 0.36 |
| CB0903 (GABA) | 3686 | 2004 | 4482 | 0.54 |

Loop-internal E→E synapses 6327 / 1851 / 6585; E→MN9 4480 / 1264 / 1675. Brain-wide median input (central brain + DNs) 422 / 207 / 714. Layer 1 keeps 0.61–0.79, loop and DNs 0.21–0.35.

The native `banc_888_synapses_v3_enriched` table shows the same: `edgelist_simple_v3` is exactly the native table with pre and post ∈ meta (Σcount 42 309 621 matches the row count; all 812/812 pre→post pairs onto MN9 and DNge062 match synapse-for-synapse), with no score, size or count filter. The per-type BANC/FAFB ratio is 0.35 with pre ∈ meta and 0.47 counting unproofread segments outside meta (22–29 % of path-cell input). Keeping the top 75/50/25 % of synapses by each dataset's own score leaves MN9 at 0.33–0.36 — the gap spans the whole score range. Caveat: the v3 table is already cut at mean_score ≳ 0.05 / size ≥ 10, a scale not comparable to FlyWire cleft ≥ 50, so we cannot see below it.

## Observation 2: labeling items

- DNge059: one cell, left only (LR_TYPE_CONFLICT / FAFB_TYPE_CONFLICT); two in FAFB and MaleCNS.
- A GABAergic right cell auto-typed `auto:DNge062` (NECK_CHECKED_NOT_AN_DN); DNge062 proper is cholinergic.
- CB0824: four cells, three transmitters (ACh/GABA/glutamate); FAFB has two, both ACh.
- Three `labellum_bristle_neuron` cells carry a `sugar*` `cell_function_detailed`.

## Observation 3: model consequence

Layer 1 responds normally in BANC (37 cells, 50–170 Hz); the loop does not ignite: CB0553 22 Hz (FAFB 69), DNge062 13 (33), CB0493 2 (44), DNge080/CB0824 0; MN9 35.6 Hz left, 0 right. Global rescaling fails: FAFB ×0.35 → MN9 0 Hz; BANC ×2 / ×2.85 / ×4 → 10.0 / 14.4 / 2.2 Hz with 5165–21043 active cells (FAFB 330). Per-type input normalization to FAFB medians (61,695 cells; factor quantiles 0.70/1.85/3.23/5.0/5.0) restores MN9 to 57.8 Hz but gives 4003 active cells, 12× FAFB; 66,903 `auto:`-prefixed cells stayed unmatched.

## Interpretation (subjective)

BANC data: 0.75 (≈0.5 detection threshold, ≈0.25 proofreading of presynaptic fragments). BANC labeling gaps: 0.12. Our artifact (transfer of the absolute per-synapse weight between datasets): 0.08. Real wiring difference: 0.05.

## Questions

1. What score threshold was applied to `synapses_v3`, and is the GNG/premotor region known to have lower detection or proofreading completeness in v888 than FAFB v783?
2. Is the `auto:` type-prefix convention documented; should `auto:X` match X across datasets?
3. Can DNge059 and `auto:DNge062` be checked?

## How we computed this

Sign rule hash f1355d3e. Code: `lab/analysis/banc_sugar_mn9.py` (paths, controls), `lab/analysis/banc_native_synapses.py` (native-table check; report `docs/2026-09-15-banc-native-synapse-check.md`), LIF `brainlab/sim/lif.py`. Runs 20260915-013801-00 (FAFB), -013810-00 (BANC), -013831-00 (MaleCNS); controls -032116-00 … -032207-00; normalized BANC -034159-00.

## What would falsify this

A BANC release or native table with a lower score cut or further proofreading in which these cells' input counts approach FAFB: if MN9 then responds, the cause was the data; if not, it moves to our model transfer or to biology.
