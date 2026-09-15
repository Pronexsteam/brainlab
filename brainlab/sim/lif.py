"""LIF в форме Shiu и др. 2024 (Brian2), на torch (CUDA, если есть). form = "shiu2024".

Уравнения, как у авторов:
    dv/dt = (v_rest − v + g)/τ_m      (unless refractory)
    dg/dt = −g/τ_syn                  (unless refractory)
    on_pre: g_post += w_syn·sign·n_syn (с задержкой delay_ms; срабатывает и в рефрактерности —
            у Brian2 «unless refractory» замораживает только дифференциальные уравнения)
    спайк: v ≥ v_th → v = v_rest, g = 0, пауза refractory_ms (v и g заморожены).
Вход kind="poisson" — в точности механизм авторов (model.py, poi()): PoissonInput с target_var='v',
то есть для каждой клетки на шаге с вероятностью rate·dt/1000 v += w_syn·f_poi (скачок в v, не в g;
68,75 мВ при константах модели — спайк в том же шаге, порог проверяется после прибавки), и у каждой Пуассон-цели рефрактерность
= 0 на весь прогон (neu[i].rfc = 0*ms у авторов; независимо от t0/t1 импульса). Поэтому Пуассон-цели
в принципе могут стрелять чаще max_rate_hz (предел для остальных клеток) — их нормированная rate
обрезается на 1.0; при 150 Гц предел не достигается. Вход kind="current": v += dt·value (мВ/мс).
Шум stimulus.noise — гауссов, прибавляется к v на активных клетках каждый шаг.

Отличие от прежней формы этого файла: раньше синаптическая переменная s входила в производную v
напрямую (v' = … + s), а не делённой на τ_m, — то есть каждый синапс действовал в τ_m = 20 раз
сильнее, чем у Shiu; g не сбрасывалась при спайке и не замораживалась в рефрактерности; dt по
умолчанию был 1 мс, теперь 0,1 мс. Константы модели (v_rest, v_th, τ_m, τ_syn, w_syn, delay_ms,
refractory_ms, g_gap, f_poi) — как в опубликованной модели. Это приведение к опубликованной
модели, а не новая модель: прежняя «диагностика» вроде «LIF молчит при g_gap 0.05» относилась
к старой форме.

Депрессия синапсов (std, необязательна): ресурс x_pre, спайк → x ← x − u·x, восстановление к 1
с τ_rec; эффективный вклад спайка = w_syn·x_pre.

Щелевые контакты (только у наборов с W_gap.nnz > 0, т.е. у червя): I_gap = g_gap·(W_gap·v − deg·v).
Интегрирование v при щелях сделано полу-неявным по диагональному члену (утечка + щелевые
контакты), а не явным Эйлером. Явная схема численно неустойчива на графе червя — у части клеток
степень по щелевым контактам доходит до 644, и dt·g_gap·deg на порядки превышает предел
устойчивости явного метода (~2) при крупном dt, из-за чего v уходит в бесконечность за десяток
шагов при первом же спайке. Используемая схема — непротиворечивая и безусловно устойчивая при
любом dt дискретизация того же ОДУ первого порядка точности:
    v_new = (v + dt·((v_rest + g)/τ_m + g_gap·W_gap·v)) / (1 + dt·a_diag) + dt·drive + noise,
    a_diag = 1/τ_m + g_gap·deg;
числитель — (v_rest + g)/τ_m, как у Shiu (g делится на τ_m), а не s напрямую; drive и шум
прибавляются отдельно, вне деления. Плата за устойчивость: при dt·g_gap·deg ≫ 1 динамика утечки
и синаптического тока у сильно щелево-связанных клеток замедляется примерно в (1 + dt·g_gap·deg)
раз — это не более грубая, а более медленная (при этом устойчивая) версия той же непрерывной
модели. При W_gap.nnz == 0 (муха) ветка не выполняется и v интегрируется явно.

Известные ограниченные отклонения от Brian2 (интегратор не меняется): (1) пришедшие спайки
прибавляются к g до интегрирования шага, поэтому эффективная синаптическая задержка —
delay = round(delay_ms/dt) = 18 шагов интегрирования против 19 у Brian2 (на один dt короче);
для прямого (feed-forward) входа это не меняет частот, только сдвигает их на 0,1 мс;
(2) явный Эйлер по v с затуханием g до шага даёт ≈ +1 % к частоте относительно точного
линейного шага (exact у Brian2) — систематическая, ограниченная ошибка.

Задержка синапсов: кольцевой буфер размером delay+1, спайк на шаге k приходит ровно
на шаге k + delay (задержка ровно delay шагов, кольцо размером delay+1).

Случайность: только torch.Generator(device).manual_seed(seed) — и Пуассон, и шум; numpy в
цикле не используется. Повторяемость байт в байт гарантируется на CPU; на CUDA проверяется
тестом test_repeat_cuda_recorded.

Нормировка rates: при refractory_ms=2.2 и dt=0.1 мс клетка не может спайковать чаще, чем
раз в round(refractory_ms/dt) = 22 шага (рефрактерность как у Brian2 ≥ 2.1: после спайка на шаге s
заморожены шаги s+1..s+21, на s+22 интегрирование снова идёт, ISI ≥ 2,2 мс; при dt = 1 мс — 2 шага),
то есть «сырая» доля шагов со спайком не превышает 1/22, и SEIZURE_LEVEL=0.9 из flags.py физически
недостижим. rates окна поэтому — не доля шагов со спайком, а эта доля, делённая на
max_rate = 1 / round(refractory_ms/dt) (максимально достижимую частоту при данных refractory_ms
и dt); 1.0 означает «клетка стреляла на пределе, разрешённом рефрактерностью», как и для плавной
модели, где r ∈ [0, 1] — предел.
В extra лежит max_rate_hz = max_rate·1000/dt; RunResult.hz(names) и rates_hz() переводят
нормированные rates обратно в герцы. Последнее окно, если duration_ms не кратно window_ms,
делится на фактическое число шагов в нём, а не всегда на per_win.
"""
import numpy as np
import torch

from . import flags, result

MODEL = "lif"


class LIF:
    def __init__(self, graph, dt_ms=0.1, v_rest=-52.0, v_th=-45.0, tau_m=20.0, tau_syn=5.0, w_syn=0.275,
                 delay_ms=1.8, refractory_ms=2.2, g_gap=0.05, f_poi=250.0, std=None, seed=0, device="auto"):
        self.g = graph
        self.params = {"dt_ms": dt_ms, "v_rest": v_rest, "v_th": v_th, "tau_m": tau_m, "tau_syn": tau_syn, "w_syn": w_syn,
                       "delay_ms": delay_ms, "refractory_ms": refractory_ms, "g_gap": g_gap, "f_poi": f_poi, "std": std,
                       "form": "shiu2024"}
        self.seed = seed
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
        c = graph.W_chem.tocoo()
        self.W = torch.sparse_coo_tensor(np.vstack([c.row, c.col]), c.data.astype(np.float32), (graph.n, graph.n)).coalesce().to(self.device)
        self.has_gap = graph.W_gap.nnz > 0
        if self.has_gap:
            gp = graph.W_gap.tocoo()
            self.G = torch.sparse_coo_tensor(np.vstack([gp.row, gp.col]), gp.data.astype(np.float32), (graph.n, graph.n)).coalesce().to(self.device)
            self.deg = torch.tensor(np.asarray(graph.W_gap.sum(axis=1)).ravel().astype(np.float32), device=self.device)
        self.v_peak_b = None      # отладка: max(v[1]) − v_rest за прогон (тест одиночного синапса)

    def run(self, stimulus, duration_ms, window_ms=50.0):
        p, n, dev = self.params, self.g.n, self.device
        gen = torch.Generator(device=dev); gen.manual_seed(self.seed)
        dt = p["dt_ms"]
        steps = int(round(duration_ms / dt)); per_win = max(1, int(round(window_ms / dt)))
        windows = int(np.ceil(steps / per_win))
        delay = max(1, int(round(p["delay_ms"] / dt))); refr = max(1, int(round(p["refractory_ms"] / dt)))
        max_rate = 1.0 / refr     # предельная частота: один спайк на round(refractory_ms/dt) шагов, как у Brian2
        decay_g = float(np.exp(-dt / p["tau_syn"]))
        pulses = stimulus.compile(self.g, dt)
        cur = [(torch.tensor(c.idx, device=dev), c.value, c.k0, c.k1) for c in pulses if c.kind == "current"]
        poi = [(torch.tensor(c.idx, device=dev), c.value * dt / 1000.0, c.k0, c.k1) for c in pulses if c.kind == "poisson"]
        v = torch.full((n,), p["v_rest"], device=dev); g = torch.zeros(n, device=dev)
        refr_left = torch.zeros(n, device=dev); x = torch.ones(n, device=dev)
        refr_n = torch.full((n,), float(refr), device=dev)       # рефрактерность по клеткам, в шагах
        for idx, _, _, _ in poi:
            refr_n[idx] = 0.0                                     # Пуассон-цели без рефрактерности (авторы: rfc = 0)
        ring = torch.zeros((delay + 1, n), device=dev)           # спайки в пути, ровно delay шагов задержки
        rates = np.zeros((windows, n), np.float32); win_acc = torch.zeros(n, device=dev); win_start = 0
        std = p["std"]; w_poi = p["w_syn"] * p["f_poi"]; v_peak = torch.tensor(-1e9, device=dev)
        if self.has_gap:
            a_diag = 1.0 / p["tau_m"] + p["g_gap"] * self.deg       # утечка + щелевые (диагональ), неявно
        for k in range(steps):
            arrived = ring[k % (delay + 1)].clone(); ring[k % (delay + 1)] = 0.0
            g = g + torch.sparse.mm(self.W, arrived.unsqueeze(1)).squeeze(1)     # on_pre — и в рефрактерности
            for idx, prob, k0, k1 in poi:
                if k0 <= k < k1:
                    v[idx] += w_poi * (torch.rand(idx.numel(), generator=gen, device=dev) < prob).float()   # target_var='v' 
            drive = torch.zeros(n, device=dev)
            for idx, val, k0, k1 in cur:
                if k0 <= k < k1:
                    drive[idx] += val
            noise = torch.randn(n, generator=gen, device=dev) * stimulus.noise if stimulus.noise else 0.0
            active = refr_left <= 0
            if self.has_gap:
                gap_off = p["g_gap"] * torch.sparse.mm(self.G, v.unsqueeze(1)).squeeze(1)   # соседи, явно
                v_new = (v + dt * ((p["v_rest"] + g) / p["tau_m"] + gap_off)) / (1.0 + dt * a_diag) + dt * drive + noise
            else:
                v_new = v + dt * (p["v_rest"] - v + g) / p["tau_m"] + dt * drive + noise
            v = torch.where(active, v_new, v)
            g = torch.where(active, g * decay_g, g)
            spike = (v >= p["v_th"]) & active
            v = torch.where(spike, torch.full_like(v, p["v_rest"]), v)
            g = torch.where(spike, torch.zeros_like(g), g)
            # refr − 1: заморожены шаги s+1..s+refr−1, интегрирование снова на s+refr (Brian2 ≥ 2.1:
            # not_refractory = timestep(t − lastspike) >= timestep(refractory)), ISI ≥ refr шагов
            refr_left = torch.where(spike, refr_n - 1.0, refr_left - 1)
            if n > 1:
                v_peak = torch.maximum(v_peak, v[1])
            eff = p["w_syn"] * (x if std else 1.0)
            ring[(k + delay) % (delay + 1)] += spike.float() * eff
            if std:
                x = (x + dt * (1.0 - x) / std["tau_rec_ms"] - std["u"] * x * spike.float()).clamp(0.0, 1.0)
            win_acc += spike.float()
            if k == steps - 1 or (k + 1) % per_win == 0:
                # переносим на CPU раз в окно, а не на каждом шаге
                rates[k // per_win] = win_acc.cpu().numpy() / (k - win_start + 1)
                win_acc.zero_(); win_start = k + 1
        self.v_peak_b = float(v_peak - p["v_rest"]) if n > 1 else None
        rates = np.clip(rates / max_rate, 0.0, 1.0)
        fl = flags.check(rates, window_ms)
        return result.RunResult(self.g.dataset, MODEL, dict(p), self.seed, stimulus.to_dict(), dt, window_ms,
                                list(self.g.names), rates, fl, result.code_hash(),
                                extra={"device": str(dev), "max_rate_hz": float(max_rate * 1000.0 / dt), "form": "shiu2024"},
                                dataset_version=self.g.version, dataset_params=dict(self.g.params))
