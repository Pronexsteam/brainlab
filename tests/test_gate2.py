import pytest

from brainlab import paths
from brainlab.lab.gates import gate2_fly_sugar as g2


def test_oracle_rates_from_parquet():
    raw = paths.DATA / "flywire_630_shiu" / "raw"
    if not (raw / "sugarR.parquet").exists():
        pytest.skip("нет sugarR.parquet")
    r = g2.oracle_rates(raw)
    assert 88 < r["720575940660219265"] < 98          # MN9 ≈ 93,3 Гц у авторов
    assert (r > 1).sum() > 100


def test_oracle_rates_reads_local_brian2_parquet():
    """Второй parquet (results/oracle_brian2/sugarR_ours.parquet, наш локальный Brian2) читается
    той же функцией с filename= — не перегоняем ворота, только проверяем чтение файла."""
    local = paths.RESULTS / "oracle_brian2" / "sugarR_ours.parquet"
    if not local.exists():
        pytest.skip("нет results/oracle_brian2/sugarR_ours.parquet")
    r = g2.oracle_rates(paths.RESULTS / "oracle_brian2", filename="sugarR_ours.parquet")
    assert (r > 1).sum() > 0


def _dataset_present():
    """Как selftest._gate2_dataset_present: кэш есть ИЛИ (база есть И dataset_info не None) —
    голого наличия store.sqlite мало, набора в ней может не быть (задача 5 финальной волны)."""
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
    """Научные ворота 2: наш LIF на данных Shiu повторяет их модель. Не ослаблять."""
    if not _dataset_present():
        pytest.skip("нет набора flywire_630_shiu")
    out = g2.run(save=False, seeds=(0,))
    print(out)
    assert out["passed"], out
