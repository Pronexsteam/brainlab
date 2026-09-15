import pyarrow as pa
import pyarrow.feather as ft
import pytest

from brainlab import paths
from brainlab.store import db, graph
from brainlab.store.loaders import fly_lee


def _toy(tmp_path):
    raw = tmp_path / "raw"; raw.mkdir()
    meta = pa.table({"fafb_783_id": ["1", "2", "3", "4"], "region": ["central_brain"] * 4,
                     "side": ["left", "right", "left", None], "flow": ["afferent", "intrinsic", "efferent", None],
                     "super_class": ["sensory", "central_brain_intrinsic", "motor", "glia"],
                     "cell_class": ["gustatory_receptor_neuron", None, None, None], "cell_sub_class": [None] * 4,
                     "cell_type": ["LB3c", "X1", "MN9", None],
                     "neurotransmitter_predicted": ["acetylcholine", "gaba", "glutamate", None],
                     "neurotransmitter_score": ["0.9", "0.8", "0.7", None],
                     "cell_function": ["gustatory", None, None, None], "cell_function_detailed": ["sugar, Gr64f", None, None, None],
                     "body_part_sensory": [None] * 4, "body_part_effector": [None] * 4})
    ft.write_feather(meta, raw / "fafb_783_meta.feather")
    edges = pa.table({"pre": ["1", "1", "2", "9"], "post": ["2", "3", "3", "1"],
                      "count": pa.array([7, 3, 12, 20], pa.int32()), "norm": [0.1] * 4, "total_input": pa.array([70, 30, 30, 5], pa.int32())})
    ft.write_feather(edges, raw / "fafb_783_simple_edgelist.feather")
    return raw


def test_load_toy(tmp_db, tmp_path):
    conn = db.connect(tmp_db)
    s = fly_lee.load(conn, _toy(tmp_path), "fafb_783", min_count=5)
    assert s["neurons"] == 4 and s["edges"] == 2 and s["skipped_edges_unknown_id"] == 1
    ns = {n["name"]: n for n in db.neurons(conn, "fafb_783")}
    assert ns["1"]["cell_class"] == "sensory" and ns["1"]["extra"]["cell_function_detailed"] == "sugar, Gr64f"
    assert ns["4"]["transmitter"] == "" and ns["2"]["side"] == "right"
    g = graph.build(conn, "fafb_783")
    assert g.W_chem[g.index["2"], g.index["1"]] == 7 and g.W_chem[g.index["3"], g.index["2"]] == -12
    assert g.W_chem[g.index["3"], g.index["1"]] == 0        # count 3 < min_count 5
    info = db.dataset_info(conn, "fafb_783")
    assert "min_count" in info["params"]
    conn.close()


def test_missing_column_is_loud(tmp_db, tmp_path):
    raw = _toy(tmp_path)
    ft.write_feather(pa.table({"a": ["1"], "b": ["2"], "count": pa.array([9], pa.int32())}), raw / "fafb_783_simple_edgelist.feather")
    conn = db.connect(tmp_db)
    with pytest.raises(ValueError, match="pre"):
        fly_lee.load(conn, raw, "fafb_783")
    conn.close()


def test_nan_string_sentinel_not_kept(tmp_db, tmp_path):
    """Регресс на находку код-ревью: Arrow-строковый столбец хранит пропуск как литеральную
    строку "NaN" (не как настоящий null), и она не должна попасть в extra как непустое значение."""
    raw = _toy(tmp_path)
    meta = ft.read_table(raw / "fafb_783_meta.feather").to_pandas()
    meta.loc[meta["fafb_783_id"] == "1", "neurotransmitter_score"] = "NaN"
    ft.write_feather(pa.Table.from_pandas(meta, preserve_index=False), raw / "fafb_783_meta.feather")
    conn = db.connect(tmp_db)
    fly_lee.load(conn, raw, "fafb_783", min_count=5)
    ns = {n["name"]: n for n in db.neurons(conn, "fafb_783")}
    assert "neurotransmitter_score" not in ns["1"]["extra"]
    assert ns["2"]["extra"]["neurotransmitter_score"] == "0.8"
    conn.close()


@pytest.mark.slow
def test_load_real_fafb(tmp_db):
    raw = paths.DATA / "fafb_783" / "raw"
    if not (raw / "fafb_783_simple_edgelist.feather").exists():
        pytest.skip("нет сырья fafb_783")
    conn = db.connect(tmp_db)
    s = fly_lee.load(conn, raw, "fafb_783")
    assert s["neurons"] == 144837 and 3_400_000 < s["edges"] < 3_600_000
    ns = {n["name"]: n for n in db.neurons(conn, "fafb_783")}
    assert ns["720575940660219265"]["cell_type"] == "MN9"
    conn.close()
