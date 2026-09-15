"""Optional run of the Shiu 2024 reference on Brian2 with their own code (data/flywire_630_shiu/raw/model.py).
Needed only to confirm that sugarR.parquet from the repository is reproducible; if Brian2 cannot be
installed on this Python — we work from sugarR.parquet and note that in the journal."""
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
    """The authors' run_exp with 21 right-side sugar GRNs; writes out_dir/sugarR_ours.parquet (if the file
    exists and force_overwrite=False — the authors' code skips the run)."""
    raw = fetch.raw_dir("flywire_630_shiu")
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)   # the authors' code writes the parquet but does not create the folder
    spec = importlib.util.spec_from_file_location("shiu_model", raw / "model.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    from brian2 import ms
    params = dict(m.default_params); params["n_run"] = n_run; params["t_run"] = t_run_ms * ms
    from .. import populations
    sugar = [int(x) for x in populations.names("flywire_630_shiu", "sugar_grn_right")]
    m.run_exp(exp_name="sugarR_ours", neu_exc=sugar, path_res=str(out_dir), path_comp=str(raw / "2023_03_23_completeness_630_final.csv"),
              path_con=str(raw / "2023_03_23_connectivity_630_final.parquet"), params=params, n_proc=n_proc, force_overwrite=force_overwrite)
    return out_dir
