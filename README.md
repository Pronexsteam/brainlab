# BrainLab

Whole-brain connectome simulation on one GPU — adult *Drosophila* (FlyWire v630 reference, FAFB v783, BANC v888, MaleCNS v0.9) and *C. elegans* — with scientific gates: every prediction is checked against published models or independent experiments before it counts.

**State (2026-09-15):** the LIF simulator (torch/CUDA) reproduces the Shiu et al. 2024 Brian2 reference on their own data (sugar → MN9 within 1.2×, Spearman 0.93 over the 200 most active cells, bitter suppression); the same taste experiment on three independent scans gives MN9 76.7 / 17.8 / 68.9 Hz — the BANC gap traces to synapse counts in the native data, reported in [htem/BANC-project#1](https://github.com/htem/BANC-project/issues/1) (full note: [docs/2026-09-15-note-banc-en.md](docs/2026-09-15-note-banc-en.md)). Raw data (~30 GB) is not in the repo; `python -m brainlab.store.fetch <dataset>` downloads it from the public buckets. Design: [docs/2026-09-14-brainlab-design.md](docs/2026-09-14-brainlab-design.md) (Russian). Built with Claude Code; MIT.

---

Симулятор поведения на реальных коннектомах (червь C. elegans, муха Drosophila) с научными
воротами: предсказания сверяются с независимыми опытами и опубликованными моделями.
Замысел — `docs/2026-09-14-brainlab-design.md`, ход работы — `docs/ЖУРНАЛ.md`.

## Сырьё и наборы

Скачать сырые файлы набора (в `data/<набор>/raw/`), затем загрузить в базу `data/store.sqlite`:

    python -m brainlab.store.fetch <набор>   # worm_cook2019, flywire_630_shiu, fafb_783, ...
    python -m brainlab.store.loaders.worm_cook2019
    python -m brainlab.store.loaders.fly_shiu630
    python -m brainlab.store.loaders.fly_lee <fafb_783|banc_888|malecns_09>

## Ворота и самопроверка

Полная самопроверка (техника + ворота 1/2 + честность прогонов):

    python selftest.py

Ворота 1 (червь) закрыты решением от 2026-09-14 — красные, не чиним. Ворота 2 (муха,
сахар/горечь против модели Shiu 2024) — по факту прогона.

## Атлас устойчивости

Один и тот же врождённый опыт на нескольких схемах мухи → `results/atlas/<имя>.{md,json}`:

    python -m brainlab.lab.atlas fly_taste

## Страница обзора

Локальный сервер, граф и проигрыватель прогонов:

    python -m brainlab.view.server   # http://127.0.0.1:8765

## Тесты

    python -m pytest -q tests          # без реальных данных, быстрые
    python -m pytest -q tests -m slow  # с реальными наборами, дольше
