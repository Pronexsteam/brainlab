# Atlas: fly_taste (2026-09-15 06:27)

| dataset | version | min_count | sign rule | baseline | sugar | sugar_bitter | valence | flags | empty groups | run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fafb_783_all | fafb_783 (lee-lab compiled_data, 2026; rule 0c66ae82) | 1 | presynaptic neurotransmitter_predicted: ach +1, gaba -1, glutamate -1, histamine -1, dopamine/serotonin/octopamine/tyramine MONOAMINE_SIGN, unclear/unknown/empty 0 | 0.0 | 70.0 | 0.0 | +1 | - | - | 20260915-062716-00 |

the model is calibrated by gate 2 on the graph with all synapses (the Shiu reference); the atlas runs on graphs with threshold min_count=5; a control run of fafb_783 at min_count=1 is a task for the next plan.

