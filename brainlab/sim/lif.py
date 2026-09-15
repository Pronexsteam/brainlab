"""LIF in the form used by Shiu et al. 2024 (Brian2), on torch (CUDA if available). form = "shiu2024".

Equations, as in the authors' model:
    dv/dt = (v_rest − v + g)/τ_m      (unless refractory)
    dg/dt = −g/τ_syn                  (unless refractory)
    on_pre: g_post += w_syn·sign·n_syn (with delay_ms delay; fires during refractoriness too —
            in Brian2 "unless refractory" only freezes the differential equations)
    spike: v ≥ v_th → v = v_rest, g = 0, refractory_ms pause (v and g frozen).
Input kind="poisson" — exactly the authors' mechanism (model.py, poi()): PoissonInput with target_var='v',
i.e. for each cell, on a step, with probability rate·dt/1000 v += w_syn·f_poi (a jump in v, not in g;
68.75 mV with the model's constants — the spike happens in the same step, the threshold is checked after
the increment), and every Poisson target has refractoriness = 0 for the whole run (neu[i].rfc = 0*ms in
the authors' code; regardless of the pulse's t0/t1). So Poisson targets can in principle fire faster than
max_rate_hz (the limit for the other cells) — their normalized rate is clipped to 1.0; at 150 Hz the limit
is not reached. Input kind="current": v += dt·value (mV/ms).
Noise stimulus.noise is Gaussian, added to v on active cells every step.

Difference from this file's previous form: previously the synaptic variable s entered the derivative of v
directly (v' = … + s), not divided by τ_m — i.e. every synapse acted τ_m = 20 times stronger than in Shiu's
model; g was not reset on spike and was not frozen during refractoriness; dt used to default to 1 ms, now
0.1 ms. The model constants (v_rest, v_th, τ_m, τ_syn, w_syn, delay_ms, refractory_ms, g_gap, f_poi) match
the published model. This is a conformance fix to the published model, not a new model: the earlier
"diagnostic" such as "LIF stays silent at g_gap 0.05" referred to the old form.

Synaptic depression (std, optional): resource x_pre, spike → x ← x − u·x, recovers to 1 with time constant
τ_rec; effective contribution of a spike = w_syn·x_pre.

Gap junctions (only for datasets with W_gap.nnz > 0, i.e. the worm): I_gap = g_gap·(W_gap·v − deg·v).
Integration of v with gap junctions is done semi-implicitly on the diagonal term (leak + gap junctions),
not with explicit Euler. The explicit scheme is numerically unstable on the worm graph — some cells reach a
gap-junction degree of up to 644, and dt·g_gap·deg exceeds the explicit method's stability limit (~2) by
orders of magnitude at a large dt, which sends v to infinity within a dozen steps on the very first spike.
The scheme used is a consistent and unconditionally stable discretization, for any dt, of the same
first-order-accurate ODE:
    v_new = (v + dt·((v_rest + g)/τ_m + g_gap·W_gap·v)) / (1 + dt·a_diag) + dt·drive + noise,
    a_diag = 1/τ_m + g_gap·deg;
the numerator is (v_rest + g)/τ_m, as in Shiu's model (g divided by τ_m), not s directly; drive and noise
are added separately, outside the division. The price of stability: when dt·g_gap·deg ≫ 1, the leak and
synaptic-current dynamics of strongly gap-coupled cells slow down by roughly a factor of
(1 + dt·g_gap·deg) — this is not a coarser but a slower (while still stable) version of the same continuous
model. When W_gap.nnz == 0 (the fly) this branch does not run and v is integrated explicitly.

Known bounded deviations from Brian2 (the integrator itself is unchanged): (1) arrived spikes are added to
g before the step is integrated, so the effective synaptic delay is
delay = round(delay_ms/dt) = 18 integration steps versus 19 in Brian2 (one dt shorter); for feed-forward
input this does not change firing rates, it only shifts them by 0.1 ms;
(2) explicit Euler on v with g decayed before the step gives ≈ +1% in firing rate relative to Brian2's
exact linear step — a systematic, bounded error.

Synaptic delay: a ring buffer of size delay+1, a spike on step k arrives exactly on step k + delay
(delay is exactly delay steps, ring size delay+1).

Randomness: only torch.Generator(device).manual_seed(seed) — for both Poisson and noise; numpy is not used
in the loop. Bit-for-bit repeatability is guaranteed on CPU; on CUDA it is checked by
test_repeat_cuda_recorded.

Normalization of rates: with refractory_ms=2.2 and dt=0.1 ms a cell cannot spike more often than once per
round(refractory_ms/dt) = 22 steps (refractoriness as in Brian2's ≥ 2.1: after a spike on step s, steps
s+1..s+21 are frozen, integration resumes on s+22, ISI ≥ 2.2 ms; with dt = 1 ms — 2 steps), i.e. the "raw"
fraction of steps with a spike never exceeds 1/22, so SEIZURE_LEVEL=0.9 from flags.py is physically
unreachable. The window rates are therefore not the fraction of steps with a spike, but that fraction
divided by max_rate = 1 / round(refractory_ms/dt) (the maximum achievable rate for the given refractory_ms
and dt); 1.0 means "the cell fired at the limit allowed by refractoriness," the same convention as for the
graded model, where r ∈ [0, 1] is the ceiling.
extra carries max_rate_hz = max_rate·1000/dt; RunResult.hz(names) and rates_hz() convert the normalized
rates back to Hz. The last window, if duration_ms is not a multiple of window_ms, is divided by the actual
number of steps it contains, not always by per_win.
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
        self.v_peak_b = None      # debug: max(v[1]) − v_rest over the run (single-synapse test)

    def run(self, stimulus, duration_ms, window_ms=50.0):
        p, n, dev = self.params, self.g.n, self.device
        gen = torch.Generator(device=dev); gen.manual_seed(self.seed)
        dt = p["dt_ms"]
        steps = int(round(duration_ms / dt)); per_win = max(1, int(round(window_ms / dt)))
        windows = int(np.ceil(steps / per_win))
        delay = max(1, int(round(p["delay_ms"] / dt))); refr = max(1, int(round(p["refractory_ms"] / dt)))
        max_rate = 1.0 / refr     # rate ceiling: one spike per round(refractory_ms/dt) steps, as in Brian2
        decay_g = float(np.exp(-dt / p["tau_syn"]))
        pulses = stimulus.compile(self.g, dt)
        cur = [(torch.tensor(c.idx, device=dev), c.value, c.k0, c.k1) for c in pulses if c.kind == "current"]
        poi = [(torch.tensor(c.idx, device=dev), c.value * dt / 1000.0, c.k0, c.k1) for c in pulses if c.kind == "poisson"]
        v = torch.full((n,), p["v_rest"], device=dev); g = torch.zeros(n, device=dev)
        refr_left = torch.zeros(n, device=dev); x = torch.ones(n, device=dev)
        refr_n = torch.full((n,), float(refr), device=dev)       # per-cell refractoriness, in steps
        for idx, _, _, _ in poi:
            refr_n[idx] = 0.0                                     # Poisson targets have no refractoriness (authors: rfc = 0)
        ring = torch.zeros((delay + 1, n), device=dev)           # spikes in transit, exactly delay steps of delay
        rates = np.zeros((windows, n), np.float32); win_acc = torch.zeros(n, device=dev); win_start = 0
        std = p["std"]; w_poi = p["w_syn"] * p["f_poi"]; v_peak = torch.tensor(-1e9, device=dev)
        if self.has_gap:
            a_diag = 1.0 / p["tau_m"] + p["g_gap"] * self.deg       # leak + gap junctions (diagonal), implicit
        for k in range(steps):
            arrived = ring[k % (delay + 1)].clone(); ring[k % (delay + 1)] = 0.0
            g = g + torch.sparse.mm(self.W, arrived.unsqueeze(1)).squeeze(1)     # on_pre — fires during refractoriness too
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
                gap_off = p["g_gap"] * torch.sparse.mm(self.G, v.unsqueeze(1)).squeeze(1)   # neighbors, explicit
                v_new = (v + dt * ((p["v_rest"] + g) / p["tau_m"] + gap_off)) / (1.0 + dt * a_diag) + dt * drive + noise
            else:
                v_new = v + dt * (p["v_rest"] - v + g) / p["tau_m"] + dt * drive + noise
            v = torch.where(active, v_new, v)
            g = torch.where(active, g * decay_g, g)
            spike = (v >= p["v_th"]) & active
            v = torch.where(spike, torch.full_like(v, p["v_rest"]), v)
            g = torch.where(spike, torch.zeros_like(g), g)
            # refr − 1: steps s+1..s+refr−1 are frozen, integration resumes on s+refr (Brian2 ≥ 2.1:
            # not_refractory = timestep(t − lastspike) >= timestep(refractory)), ISI ≥ refr steps
            refr_left = torch.where(spike, refr_n - 1.0, refr_left - 1)
            if n > 1:
                v_peak = torch.maximum(v_peak, v[1])
            eff = p["w_syn"] * (x if std else 1.0)
            ring[(k + delay) % (delay + 1)] += spike.float() * eff
            if std:
                x = (x + dt * (1.0 - x) / std["tau_rec_ms"] - std["u"] * x * spike.float()).clamp(0.0, 1.0)
            win_acc += spike.float()
            if k == steps - 1 or (k + 1) % per_win == 0:
                # move to CPU once per window, not on every step
                rates[k // per_win] = win_acc.cpu().numpy() / (k - win_start + 1)
                win_acc.zero_(); win_start = k + 1
        self.v_peak_b = float(v_peak - p["v_rest"]) if n > 1 else None
        rates = np.clip(rates / max_rate, 0.0, 1.0)
        fl = flags.check(rates, window_ms)
        return result.RunResult(self.g.dataset, MODEL, dict(p), self.seed, stimulus.to_dict(), dt, window_ms,
                                list(self.g.names), rates, fl, result.code_hash(),
                                extra={"device": str(dev), "max_rate_hz": float(max_rate * 1000.0 / dt), "form": "shiu2024"},
                                dataset_version=self.g.version, dataset_params=dict(self.g.params))
