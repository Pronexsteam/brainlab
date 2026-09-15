# Атлас: fly_taste (2026-09-15 03:41)

| набор | версия | min_count | правило знаков | baseline | sugar | sugar_bitter | оценка | флаги | пустые группы | прогон |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| banc_888_norm | banc_888 v3 (lee-lab compiled_data, 2026; rule f1355d3e) norm:fafb_783 | 5 | presynaptic neurotransmitter_predicted: ach +1, gaba -1, glutamate -1, histamine -1, dopamine/serotonin/octopamine/tyramine MONOAMINE_SIGN, unclear/unknown/empty 0 | 0.0 | 57.8 | 0.0 | +1 | - | - | 20260915-034159-00 |

модель откалибрована воротами 2 на графе со всеми синапсами (эталон Shiu); атлас идёт на графах с порогом min_count=5; контрольный прогон fafb_783 при min_count=1 — задача следующего плана.

