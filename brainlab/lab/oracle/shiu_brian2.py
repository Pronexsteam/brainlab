"""Необязательный прогон эталона Shiu 2024 на Brian2 их же кодом (data/flywire_630_shiu/raw/model.py).
Нужен только чтобы подтвердить, что sugarR.parquet из репозитория воспроизводится; если Brian2 не
ставится на этот Python — работаем по sugarR.parquet и записываем это в журнал."""
import importlib.util
from pathlib import Path

from ...store import fetch


def available():
    try:
        import brian2  # noqa: F401
        return True
    except Exception:
        return False


def run_sugar(out_dir, n_run=3, t_run_ms=1000, n_proc=1, force_overwrite=False):
    """Авторский run_exp с 21 сахарной GRN справа; пишет out_dir/sugarR_ours.parquet (если файл есть и
    force_overwrite=False — авторский код пропускает прогон)."""
    raw = fetch.raw_dir("flywire_630_shiu")
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)   # авторский код пишет parquet, папку не создаёт
    spec = importlib.util.spec_from_file_location("shiu_model", raw / "model.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    from brian2 import ms
    params = dict(m.default_params); params["n_run"] = n_run; params["t_run"] = t_run_ms * ms
    from .. import populations
    sugar = [int(x) for x in populations.names("flywire_630_shiu", "sugar_grn_right")]
    m.run_exp(exp_name="sugarR_ours", neu_exc=sugar, path_res=str(out_dir), path_comp=str(raw / "2023_03_23_completeness_630_final.csv"),
              path_con=str(raw / "2023_03_23_connectivity_630_final.parquet"), params=params, n_proc=n_proc, force_overwrite=force_overwrite)
    return out_dir
