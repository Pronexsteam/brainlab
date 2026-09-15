import pytest

from brainlab import paths
from brainlab.lab.gates import gate2_fly_sugar as g2


def test_oracle_rates_from_parquet():
    raw = paths.DATA / "flywire_630_shiu" / "raw"
    if not (raw / "sugarR.parquet").exists():
        pytest.skip("no sugarR.parquet")
    r = g2.oracle_rates(raw)
    assert 88 < r["720575940660219265"] < 98          # MN9 ≈ 93.3 Hz in the authors' data
    assert (r > 1).sum() > 100


def test_oracle_rates_reads_local_brian2_parquet():
    """The second parquet (results/oracle_brian2/sugarR_ours.parquet, our local Brian2) is read
    by the same function with filename= — we don't rerun the gate, only check reading the file."""
    local = paths.RESULTS / "oracle_brian2" / "sugarR_ours.parquet"
    if not local.exists():
        pytest.skip("no results/oracle_brian2/sugarR_ours.parquet")
    r = g2.oracle_rates(paths.RESULTS / "oracle_brian2", filename="sugarR_ours.parquet")
    assert (r > 1).sum() > 0


def _dataset_present():
    """Like selftest._gate2_dataset_present: cache present OR (database present AND dataset_info not None) —
    store.sqlite existing alone is not enough, the dataset may not be in it (task 5 of the final wave)."""
    if (paths.CACHE / "flywire_630_shiu.npz").exists():
        return True
    if not paths.DB_PATH.exists():
        return False
    from brainlab.store import db
    conn = db.connect()
    try:
        return db.dataset_info(conn, "flywire_630_shiu") is not None
    finally:
        conn.close()


def test_gate2_passes():
    """Scientific gate 2: our LIF reproduces their model on the Shiu data. Do not weaken."""
    if not _dataset_present():
        pytest.skip("no flywire_630_shiu dataset")
    out = g2.run(save=False, seeds=(0,))
    print(out)
    assert out["passed"], out
