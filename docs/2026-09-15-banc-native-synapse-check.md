# Check: is the BANC synapse deficit in the native table, or only in the Lee-lab edgelist?

Question from `docs/2026-09-15-banc-sugar-mn9-analysis.md` (§4, §7) and `2026-09-15-note-banc.md` ("what would falsify this"):
on the sugar → MN9 path cells in BANC v888, per `banc_888_edgelist_simple_v3.feather` the inputs are ≈0.39× FAFB v783 (MN9 2816 vs 8021,
DNge062 679 / 2560, DNge080 879 / 4254). Is this also visible in the native synapse table, or is it an artefact of how the Lee-lab edgelist was compiled (a score cut)?

Script: `lab/analysis/banc_native_synapses.py` (one pass over `banc_888_synapses_v3_enriched.parquet`, 198,816,365 rows, 1989 row groups,
4 columns, ≈4 min; over `fafb_783_synapses.parquet`, 55,595,125 rows, ≈2.5 min; histogram cache in a temp folder, `--stage report` for the report only).
All numbers below are from its output. Path cells are taken from `data/store.sqlite` (BANC: cell_type or extra.fafb_cell_type exactly equal to the type, no `auto:`).

## Summary

1. **The BANC edgelist = the native table with no independent threshold**, only the restriction "pre and post are among the 188,508 meta cells." Native rows
   198,816,365; with post ∈ meta 57,003,879 (28.7 %); with pre and post ∈ meta **42,309,621 = edgelist Σcount 42,309,621** exactly. By cell:
   input matches exactly for 100 % of cells (167,018); by pre→post pair for MN9 and DNge062 — 812 of 812 pairs match synapse-for-synapse.
   There is no filter on mean_score, size, or synapse count per pair (7.37 million pairs with count = 1) in the edgelist.
2. **The deficit is present in the native data.** With the same restriction (pre ∈ meta), BANC/FAFB per type is 0.21–0.61, median **0.35**; if
   BANC includes all inputs, including from segments outside meta (unproofread fragments, 22–29 % of path-cell input) — 0.28–0.83, median **0.47**.
   Even under the most generous count, MN9 has 3822 inputs vs 7753, DNge080 1155 vs 4149.
3. **A score cut is not the cause, even on a matched filter**: keeping the top 75 / 50 / 25 % of synapses by each dataset's own score leaves the
   ratio unchanged (MN9 0.36 → 0.36 → 0.33 → 0.32; DNge080 0.21 → 0.21 → 0.20 → 0.20). The deficit spans the whole score distribution, not the tail.
4. Caveat: the native v3 table itself is already cut before it reaches us — mean_score ≥ ~0.05 (a soft edge: 20 rows in the 0.051 bin, 320,470 in the 0.062 bin,
   5th percentile 0.072, median 0.118) and size ≥ 10 (median 41). What was dropped below 0.05 is not visible from this file; the score
   scale is not comparable to the FlyWire cleft ≥ 50 convention. The FAFB parquet is likewise already filtered (confidence min 51, a hard edge: 158,106 rows at 51).
5. **Conclusion for the note:** the cause remains on the BANC data side (synapse detection at their threshold and/or incomplete proofreading of
   presynaptic fragments), not in how the Lee-lab edgelist was compiled. Send the note to the BANC team; the "refutation via the native table" branch is closed.

## 1. The whole file

| | BANC v888 (synapses_v3_enriched) | FAFB v783 (synapses.parquet) |
| --- | --- | --- |
| native rows | 198,816,365 | 55,595,125 |
| post ∈ meta | 57,003,879 (28.7 %) | 55,595,125 (100 %) |
| pre and post ∈ meta | **42,309,621** (21.3 %) | 55,595,125 (100 %) |
| edgelist Σcount | **42,309,621** (100.0 % of pre,post ∈ meta) | 68,912,106 (124 % of parquet) |
| edgelist: rows / min count / pairs with count=1 | 13,620,865 / 1 / 7,370,440 | 15,023,799 / 1 / 6,127,757 (59 duplicate pairs) |
| score: min / quantiles 5–25–50–75–95 | 0.051 / 0.072–0.095–0.118–0.144–0.186 | 51 / 66–125–141–146–161 |
| size (BANC): min / quantiles | 10 / 12–25–41–65–118 | — |
| score threshold reproducing edgelist Σ among "all" | mean_score ≥ 0.156 → 41.7 M (Δ −0.6 M) | none (edgelist exceeds parquet) |
| score threshold among "pre,post ∈ meta" | **≥ 0 → 42,309,621 (Δ 0)** | none |

Per-cell reconciliation (edgelist input per cell vs native input from pre ∈ meta at a score threshold):

| set | threshold | cells | exact match | median edgelist/native |
| --- | --- | --- | --- | --- |
| BANC | 0 (no threshold) | 167,018 | **100.0 %** | 1.000 |
| BANC | mean_score ≥ 0.08 | 167,018 | 3.5 % | 1.126 |
| BANC | ≥ 0.10 | 167,018 | 1.2 % | 1.449 |
| BANC | ≥ 0.15 | 167,018 | 0.2 % | 4.87 |
| BANC, all pre (incl. outside meta) | 0 | 170,614 | 1.5 % | 0.716 |
| FAFB | 0 | 136,594 | 7.7 % | 1.122 |
| FAFB | confidence ≥ 60 | 136,583 | 4.4 % | 1.157 |

BANC: edgelist_simple_v3 is exactly the native v3 table restricted to pairs of meta cells. FAFB: the Lee-lab parquet is a subset of the source
that the edgelist was compiled from (24 % of synapses are missing; for path cells 3.6 %: 80 of 1035 pairs onto MN9/DNge062 are missing entirely,
i.e. some presynaptic cells dropped out, 764 synapses out of 21,161, while the pairs that are present match 955 of 955). The FlyWire cleft ≥ 50
convention is visible at the lower edge (min 51), but the edgelist is not reproduced from the parquet — for FAFB the edgelist remains the reference,
with a ≤5 % difference on path cells.

Median input per cell (central_brain, cells with ≥1 native input): BANC (43,585 cells) all pre 212, pre ∈ meta = edgelist 145;
FAFB (37,676) parquet 264, edgelist 328. BANC/FAFB ratio 0.44 by edgelist, 0.80 if BANC with all pre is compared to FAFB parquet
(in the 2026-09-15 analysis it was 207 / 422 = 0.49 at min_count 5 and central_brain + descending).

## 2. Path cells

BANC (native_all — all inputs; pre ∈ meta = edgelist, 100 % match; meta_input — the Lee-lab meta `input_connections` column):

| type | side | native_all | pre ∈ meta = edgelist | meta_input | score q50 | size q50 |
| --- | --- | --- | --- | --- | --- | --- |
| MN9 | l | 3259 | 2446 | 2937 | 0.121 | 50 |
| MN9 | r | 4386 | 3186 | 3842 | 0.118 | 49 |
| DNge062 | l | 838 | 643 | 558 | 0.118 | 36 |
| DNge062 | r | 916 | 715 | 609 | 0.115 | 35 |
| DNge080 | l | 1318 | 1012 | 1049 | 0.122 | 35 |
| DNge080 | r | 992 | 746 | 805 | 0.122 | 35 |
| DNge059 | l (only one) | 3080 | 2330 | 2351 | 0.118 | 42 |
| CB0553 | l | 3324 | 2664 | 2656 | 0.119 | 45 |
| CB0553 | r | 3101 | 2376 | 2802 | 0.119 | 45 |
| CB0493 | l | 2153 | 1704 | 1702 | 0.117 | 45 |
| CB0493 | r | 2251 | 1785 | 0 (!) | 0.118 | 46 |
| CB0824 | l, l, l, r | 1085 / 562 / 899 / 1478 | 866 / 399 / 693 / 1154 | 846 / 358 / 639 / 1276 | 0.114–0.118 | 35–37 |
| CB0393 | l / r | 3336 / 3893 | 2400 / 3171 | 0 (!) / 3332 | 0.119 / 0.122 | 42 / 46 |
| CB0616 | l / r | 812 / 814 | 542 / 618 | 527 / 695 | 0.117 / 0.119 | 43 / 44 |

All path cells in BANC are proofread = TRUE; DNge062/DNge080/DNge059 carry status REVIEW_MATCH_AN_DN; MN9 — HAS_MANUAL_ANNOTATION. The median
input score of path cells (0.115–0.122) equals the brain-wide median (0.118) — synapses on these cells are not "worse" than elsewhere. The
meta `input_connections` column for CB0393 (l) and CB0493 (r) = 0 against 2400 / 1785 native inputs — a meta glitch, already noted in the analysis.

FAFB (parquet vs edgelist): MN9 7979 / 7528 vs 8174 / 7868; DNge062 2588 / 2302 vs 2690 / 2429; DNge080 4431 / 3867 vs 4516 / 3991;
DNge059 7636 / 6962 vs 7880 / 7161; CB0553 6133 / 5102 vs 6191 / 5146; CB0493 5213 / 4654 vs 5422 / 4790; CB0824 2108 / 2152 vs
2153 / 2178; CB0393 5355 / 3381 vs 5744 / 3432; CB0616 1542 / 1282 vs 1615 / 1349 (edgelist/parquet 1.01–1.05); median confidence 141–143.

## 3. BANC vs FAFB by type (mean per cell of the type)

| type | cells B / F | BANC native, all pre | BANC edgelist (= pre ∈ meta) | FAFB parquet | FAFB edgelist | **native B/F** | **edgelist B/F** | BANC edgelist/native |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MN9 | 2 / 2 | 3822 | 2816 | 7753 | 8021 | 0.49 | 0.35 | 0.74 |
| DNge062 | 2 / 2 | 877 | 679 | 2445 | 2559 | 0.36 | 0.27 | 0.77 |
| DNge080 | 2 / 2 | 1155 | 879 | 4149 | 4253 | 0.28 | 0.21 | 0.76 |
| DNge059 | 1 / 2 | 3080 | 2330 | 7299 | 7520 | 0.42 | 0.31 | 0.76 |
| CB0553 | 2 / 2 | 3212 | 2520 | 5617 | 5668 | 0.57 | 0.44 | 0.78 |
| CB0493 | 2 / 2 | 2202 | 1744 | 4933 | 5106 | 0.45 | 0.34 | 0.79 |
| CB0824 | 4 / 2 | 1006 | 778 | 2130 | 2165 | 0.47 | 0.36 | 0.77 |
| CB0393 | 2 / 2 | 3614 | 2785 | 4368 | 4588 | 0.83 | 0.61 | 0.77 |
| CB0616 | 2 / 2 | 813 | 580 | 1412 | 1482 | 0.58 | 0.39 | 0.71 |
| **median** | | | | | | **0.47** | **0.35** | 0.77 |

"native B/F" is the most generous estimate for BANC (all presynaptic segments, including unproofread fragments, against the FAFB parquet, which
has no such segments). "edgelist B/F" is what our graph actually sees; the median 0.35 matches the 0.39 from the analysis (there 19 types, here 9).

Matched filter — in each set, the top X % of synapses (pre,post ∈ meta) by its own score (threshold BANC / FAFB: 100 % — 0 / 0; 75 % — 0.095 / 125;
50 % — 0.118 / 141; 25 % — 0.144 / 146), BANC/FAFB ratio:

| type | 100 % | 75 % | 50 % | 25 % |
| --- | --- | --- | --- | --- |
| MN9 | 0.36 | 0.36 | 0.33 | 0.32 |
| DNge062 | 0.28 | 0.26 | 0.26 | 0.27 |
| DNge080 | 0.21 | 0.21 | 0.20 | 0.20 |
| DNge059 | 0.32 | 0.31 | 0.28 | 0.27 |
| CB0553 | 0.45 | 0.42 | 0.38 | 0.37 |
| CB0493 | 0.35 | 0.34 | 0.30 | 0.30 |
| CB0824 | 0.37 | 0.35 | 0.32 | 0.32 |
| CB0393 | 0.64 | 0.62 | 0.57 | 0.54 |
| CB0616 | 0.41 | 0.40 | 0.38 | 0.33 |

The gap does not depend on where the score cut is made — so it is not in the detector's confidence distribution within the available range, but
in the raw number of detected synapses itself.

## 4. Pre→post pairs (MN9 + DNge062, pre ∈ meta)

BANC: 812 pairs, matching the edgelist 812 / 812 (Σ 6990 = 6990); largest DNge062→MN9 230 and 190, CB0465→MN9 221 and 212, CB0553→MN9 191 and 176.
FAFB: 955 pairs from the parquet — 955 / 955 (Σ 20,397); DNge062→MN9 900 and 841, CB0553→MN9 691 and 619, CB0465→MN9 684 and 599. The same
edges, 3–4.5× fewer synapses in BANC, and this is not a result of the threshold.

## 5. Verdict

- **The 0.35–0.47 deficit is present in the native BANC v3 table.** The Lee-lab edgelist introduces no threshold: it is the native table reduced
  to pairs of meta cells (42,309,621 synapses exactly, 100 % of cells and pairs match). The "edgelist compilation" hypothesis is closed.
- Part of the gap comes from **unproofread presynaptic fragments**: 22–29 % of path-cell input in BANC comes from segments not among the
  188,508 meta cells (brain-wide: median central_brain input 212 vs 145). Both the edgelist and our graph drop these; but even counting them, BANC/FAFB is
  0.47 (MN9 0.49, DNge080 0.28). This is a sign of incomplete proofreading/assembly (orphan axons), not of a threshold.
- The remainder (≈2× at all-pre) is **synapse detection** in BANC, including a threshold applied by the BANC team themselves before releasing
  table v3 (mean_score ≥ ~0.05, size ≥ 10): we cannot see below it, and the score scale is not comparable to FlyWire cleft ≥ 50. Within the
  available range the ratio is stable (0.32–0.36 for MN9 at any quantile), so a "hard detector" would explain the gap only if the missed synapses
  lay entirely below 0.05 — this cannot be checked without the raw BANC table.
- Probabilities from the analysis (0.70 detection / 0.15 annotation / 0.10 our artefact / 0.05 biology) → **0.75 BANC data** (of which ≈0.5 detection
  threshold, ≈0.25 incomplete proofreading of presynaptic fragments) / **0.12 annotation** (DNge059 on the right, auto:DNge062, CB0824 — unchanged)
  / **0.08 ours** (edgelist compilation ruled out; what remains is transferring Shiu's absolute weight between scans) / **0.05 biology**.
- **Note:** addressee is the BANC team (and, for information, Lee-lab as the bucket owner), not "a question to Lee-lab about a filter." Remove the
  item "what would falsify this — confirmation that the low numbers are an intentional edgelist_simple_v3 filter" from both notes: checked, there
  is no filter. Add: (a) the numbers were cross-checked against the native `synapses_v3`: edgelist = native table ∩ meta; (b) 22–29 % of path-cell
  input comes from segments outside meta; (c) questions for BANC: what threshold was applied to table v3 (mean_score / size) and is it comparable
  to FlyWire cleft ≥ 50; is proofreading of presynaptic axons in the GNG/premotor region complete in v888; is there a recall estimate for the
  synapse detector in this region. Manual cross-check in BANC Codex (as noted, "what to do about it") is no longer needed as a condition for
  sending — it has been replaced by this check.
