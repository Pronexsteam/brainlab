# BrainLab

**Finding (2026-09-15):** a published whole-brain model of the fly (Shiu et al. 2024, LIF with an absolute weight of 0.275 mV per synapse) depends on the *absolute synapse count* of the scan it was tuned on. Run unchanged on three independent adult *Drosophila* connectomes, the same sugar → proboscis-motor-neuron (MN9) experiment gives **76.7 Hz (FAFB v783), 68.9 Hz (MaleCNS v0.9) and 17.8 Hz (BANC v888)**. The path is the same in all three; BANC simply carries ~0.35× the synapses on the same identified cells, and that deficit is present in the native BANC synapse table, not introduced by any compilation step. Global rescaling does not fix it (inhibition rises first); per-type input normalization restores MN9 but inflates brain-wide activity 12×. Anyone comparing connectomes with an absolute-weight model needs to control for this. Details: [docs/2026-09-15-note-banc-en.md](docs/2026-09-15-note-banc-en.md); reported to the data team in [htem/BANC-project#1](https://github.com/htem/BANC-project/issues/1).

**What this is:** a small, single-GPU simulator of real connectomes (adult fly: FlyWire v630 reference, FAFB v783, BANC v888, MaleCNS v0.9; worm: *C. elegans* Cook 2019) with *gates* — nothing counts as a result until it reproduces a published model or an independent experiment. Every run is stored with dataset version, dataset parameters (sign rule, synapse threshold), model parameters, seed and code hash. Runs are never deleted.

**What this is not:** a discovery about how the fly thinks. Everything here is a model prediction; it becomes a fact only when someone with living flies tests it. Gate 1 (worm touch response) is red and left red on purpose — the worm connectome lacks the physiology to pass it, and we did not tune it.

Built in two nights with Claude Code. Code: MIT. Data: see [Data and citations](#data-and-citations) — not included in the repo, downloaded by the fetch script.

## Reproduce the numbers

Python 3.14, `torch` with CUDA (CPU works, ~50× slower), `numpy scipy pandas pyarrow pyyaml networkx pytest playwright`. From the repo root:

```bash
# 1. reference data (~100 MB): worm (OpenWorm c302) and the Shiu 2024 FlyWire v630 files
python -m brainlab.store.fetch worm flywire_630_shiu
python -m brainlab.store.loaders.worm_cook2019
python -m brainlab.store.loaders.fly_shiu630          # ~1 min, 14.7 M synapses into data/store.sqlite

# 2. the self-test: unit tests, gate 1 (worm, expected red), gate 2 (fly vs Shiu 2024), honesty checks
python selftest.py
```

Expected gate-2 line on our machine (RTX 5050, seed 0; Poisson input is sampled, so expect ±5 % on rates):

```
gate 2: {"mn9_ours_hz": 111.0, "mn9_oracle_hz": 93.27, "ratio": 1.19, "spearman_top200": 0.926,
           "active_ours": 372, "active_oracle": 389, "suppression": 0.018, "mn9_local_brian2_hz": 82.67, "passed": true}
gate 1 (worm, closed by the 2026-09-14 decision): red
gate 2 (fly): green
VERDICT: RED            # red because gate 1 is red — by design
```

Criteria for gate 2 (fixed before the first run, never changed): 0.6 ≤ MN9 ratio ≤ 1.4 vs the authors' shipped `sugarR.parquet`; Spearman ≥ 0.7 over their 200 most active cells; active-cell count within 0.7–1.4×; sugar+bitter ≤ 0.5 × sugar. If `brian2` is installed, `brainlab/lab/oracle/shiu_brian2.py` also runs the authors' own code (we get MN9 82.7 Hz from it vs 93.3 Hz in their shipped file; our 111 Hz is 1.35× the local Brian2 run — inside the gate, but the margin is on that side).

```bash
# 3. the three-scan atlas (~4 GB download, ~1 h to load MaleCNS, ~1 min on GPU to run)
python -m brainlab.store.fetch fafb_783 banc_888 malecns_09
python -m brainlab.store.loaders.fly_lee fafb_783 banc_888 malecns_09
python -m brainlab.lab.atlas fly_taste                 # → results/atlas/fly_taste.md
python -m brainlab.lab.atlas fly_taste banc_888_norm   # BANC with per-type input normalization → 57.8 Hz
```

Expected `results/atlas/fly_taste.md`: MN9 sugar / sugar+bitter = fafb_783 76.7 / 0.0, banc_888 17.8 / 4.4, malecns_09 68.9 / 0.0 Hz (our table is committed in `results/atlas/`).

```bash
# 4. the synapse-level check behind the BANC finding (native synapse tables: 2 GB + 20 GB)
python -m brainlab.store.fetch --extra fafb_783 banc_888
python lab/analysis/banc_native_synapses.py            # ~6 min → docs/2026-09-15-проверка-синапсов-banc.md
```

Other commands: `python -m brainlab.view.server` (local viewer, http://127.0.0.1:8765, subgraph by cell group + state panel), `python -m pytest -q tests` (fast; `-m slow` for real-data loader tests).

## Assumptions stated out loud

- Synapse sign from the predicted presynaptic transmitter: ACh +1, GABA −1, glutamate −1, histamine −1, dopamine/serotonin/octopamine/tyramine +1 (verified 99.97 % against the reference's own signs), unknown 0. Recorded per dataset as `sign_rule`; a parameter, not a truth.
- Lee-lab compiled edgelists are stored with ≥ 5 synapses per pair (Codex convention); the Shiu reference keeps every synapse. The model was validated on the latter and the atlas runs on the former — a control run of FAFB at threshold 1 is the next item.
- Taste groups are labellar GRNs only (as in the reference); BANC and MaleCNS also contain leg and wing GRNs, kept as `*_all` groups.
- The LIF differs from Brian2 by one integration step in synaptic delay (18 vs 19 steps at 0.1 ms) and uses explicit Euler for v (≈ +1 % rates); documented in `brainlab/sim/lif.py`.

## Layout

`brainlab/store` (SQLite + sparse cache, fetch, loaders) · `brainlab/sim` (graded model, LIF, stimulus, run result) · `brainlab/lab` (cell populations by rules, state vector, YAML experiment runner, gates, atlas, analyses) · `brainlab/view` (stdlib server + canvas page) · `lab/` (experiments, populations, analyses) · `docs/` (design, survey, plans, journal — Russian) · `results/atlas`, `results/screens`.

## Data and citations

Raw data is **not** in this repository. `brainlab/store/fetch.py` downloads it from the owners' public locations; please cite them, and respect their licenses:

| Dataset | Source | License | Cite |
| --- | --- | --- | --- |
| FlyWire / FAFB v783 (`fafb_783`) | Lee-lab `compiled_data` bucket (from FlyWire) | **CC BY-NC 4.0** (non-commercial) | Dorkenwald et al., *Nature* 634, 124–138 (2024); Schlegel et al., *Nature* 634, 139–152 (2024) |
| FlyWire v630 as used by Shiu et al. (`flywire_630_shiu`) | github.com/philshiu/Drosophila_brain_model | code MIT; data derived from FlyWire, **CC BY-NC 4.0** | Shiu et al., *Nature* (2024), "A Drosophila computational brain model reveals sensorimotor processing" |
| BANC v888 (`banc_888`) | Lee-lab `compiled_data` bucket; Harvard Dataverse doi:10.7910/DVN/7WTH1N | CC BY 4.0 | Bates, Phelps, Kim, Yang et al., *Nature* (2026), "Distributed control circuits across a brain-and-cord connectome" |
| MaleCNS v0.9 (`malecns_09`) | Lee-lab `compiled_data` bucket (from Janelia FlyEM) | CC BY 4.0 | Janelia FlyEM Male CNS connectome (male-cns.janelia.org) |
| *C. elegans* Cook 2019 (`worm_cook2019`) | OpenWorm c302 repository | c302 MIT; data from Cook et al. | Cook et al., *Nature* 571, 63–71 (2019) |
| Compiled edgelists and metadata for the three fly scans | github.com/sjcabs/fly_connectome_data_tutorial (A. S. Bates) | see repository | cite the tutorial and the underlying datasets |

Because two of the datasets are CC BY-NC, anything built on FAFB/FlyWire-derived files here is non-commercial. Check each owner's page for current terms before reuse.
