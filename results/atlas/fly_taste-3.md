# Atlas: fly_taste (2026-09-15 06:31)

| dataset | version | min_count | sign rule | baseline | sugar | sugar_bitter | valence | flags | empty groups | run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| banc_888_norm2 | banc_888 v3 (lee-lab compiled_data, 2026; rule f1355d3e) norm:fafb_783 | 5 | presynaptic neurotransmitter_predicted: ach +1, gaba -1, glutamate -1, histamine -1, dopamine/serotonin/octopamine/tyramine MONOAMINE_SIGN, unclear/unknown/empty 0 | 0.0 | 44.4 | 0.0 | +1 | - | - | 20260915-063110-00 |

the model is calibrated by gate 2 on the graph with all synapses (the Shiu reference); the atlas runs on graphs with threshold min_count=5; the control run fafb_783_all (min_count=1, 15.0 M edges) gives MN9 sugar 70.0 Hz / sugar+bitter 0.0 Hz, 344 active cells, against 76.7 / 0.0 / 330 at min_count=5.

