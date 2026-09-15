"""Input normalization by cell type (carrying the Shiu model between scans): toy graphs."""
import numpy as np
import pytest
import scipy.sparse as sp

from brainlab import paths
from brainlab.store import graph


def _toy(names, chem, gap=(), dataset="toy", version="v1"):
    n = len(names)
    idx = {x: i for i, x in enumerate(names)}
    rc, cc, vc = zip(*[(idx[post], idx[pre], w) for pre, post, w in chem]) if chem else ([], [], [])
    rg, cg, vg = zip(*[(idx[post], idx[pre], w) for pre, post, w in gap]) if gap else ([], [], [])
    Wc = sp.csr_matrix((np.array(vc, dtype=np.float32), (rc, cc)), shape=(n, n), dtype=np.float32)
    Wg = sp.csr_matrix((np.array(vg, dtype=np.float32), (rg, cg)), shape=(n, n), dtype=np.float32)
    return graph.Graph(dataset, list(names), [""] * n, [""] * n, Wc, Wg, version=version, params={"min_count": 5})


def _pair():
    # reference graph: three type-A cells with inputs 10, 20, 30 -> median 20; src is a source with no inputs
    ref = _toy(["src", "a1", "a2", "a3"], [("src", "a1", 10), ("src", "a2", -20), ("src", "a3", 30)], dataset="ref")
    ref_types = ["", "A", "A", "A"]
    # ds: a type-A cell with input 5 (-> x4), a type-A cell with input 200 (-> clipped to 0.2), no type (-> 1.0),
    # type B which is absent from ref (-> 1.0); gap junction lo-hi
    ds = _toy(["src", "lo", "hi", "notype", "b"],
              [("src", "lo", 5), ("src", "hi", -200), ("src", "notype", 7), ("src", "b", 3)],
              gap=[("lo", "hi", 2), ("hi", "lo", 2)], dataset="ds", version="v9")
    ds_types = ["", "A", "A", "", "B"]
    return ds, ds_types, ref, ref_types


def test_normalize_factors_and_clip():
    ds, ds_types, ref, ref_types = _pair()
    Wg_before = ds.W_gap.copy()
    g = graph.normalize_inputs(ds, ref, ds_types, ref_types)
    row = lambda name: g.W_chem[g.index[name]].toarray().ravel()
    assert row("lo")[g.index["src"]] == pytest.approx(20.0)            # 5 x 4.0
    assert row("hi")[g.index["src"]] == pytest.approx(-200 * 0.2)      # 20/200=0.1 -> clipped to 0.2
    assert row("notype")[g.index["src"]] == pytest.approx(7.0)         # no type — 1.0
    assert row("b")[g.index["src"]] == pytest.approx(3.0)              # type absent from ref — 1.0
    assert (g.W_gap != Wg_before).nnz == 0                             # W_gap is untouched
    assert (ds.W_chem[ds.index["lo"]].toarray().ravel()[ds.index["src"]]) == 5.0   # the original graph is unchanged
    assert g.dataset == "ds" and g.version == "v9 norm:ref" and g.names == ds.names
    p = g.params["normalize"]
    assert p["ref"] == "ref" and p["factor_min"] == 0.2 and p["factor_max"] == 5.0
    assert p["cells_scaled"] == 2 and p["cells_unscaled"] == 3
    assert p["factor_median"] == pytest.approx(1.0)                    # median over all cells: [1,1,1,0.2,4] -> 1
    assert "rule" in p and g.params["min_count"] == 5                  # old params are preserved
    assert p["median_over"] == "positive" and p["strip_auto"] is False and p["regions"] is None


def test_median_ignores_zero_input_ref_cells():
    """The type target is the median over reference cells with in_ref > 0: a type-A cell with no
    inputs at all (unreconstructed) must not pull the target down."""
    ref = _toy(["src", "a1", "a2", "a3", "a0"], [("src", "a1", 10), ("src", "a2", 20), ("src", "a3", 30)], dataset="ref")
    ref_types = ["", "A", "A", "A", "A"]                                # a0 has input 0
    ds = _toy(["src", "x"], [("src", "x", 5)], dataset="ds")
    g = graph.normalize_inputs(ds, ref, ["", "A"], ref_types)
    assert g.W_chem[g.index["x"], g.index["src"]] == pytest.approx(20.0)   # median of (10, 20, 30), not of (0, 10, 20, 30)


def test_strip_auto_matches_prefixed_type():
    ds, _, ref, ref_types = _pair()
    ds_types = ["", "auto:A", "A", "", "B"]
    g0 = graph.normalize_inputs(ds, ref, ds_types, ref_types)
    assert g0.W_chem[g0.index["lo"], g0.index["src"]] == pytest.approx(5.0)     # auto:A is not A by string
    assert g0.params["normalize"]["auto_matched"] == 0
    g = graph.normalize_inputs(ds, ref, ds_types, ref_types, strip_auto=True)
    assert g.W_chem[g.index["lo"], g.index["src"]] == pytest.approx(20.0)      # auto:A -> A -> x4
    p = g.params["normalize"]
    assert p["strip_auto"] is True and p["auto_matched"] == 1 and p["cells_scaled"] == 2


def test_regions_restrict_scaling():
    ds, ds_types, ref, ref_types = _pair()
    cell_regions = ["central_brain", "central_brain", "optic_lobe", "central_brain", ""]
    g = graph.normalize_inputs(ds, ref, ds_types, ref_types, regions=["central_brain"], cell_regions=cell_regions)
    assert g.W_chem[g.index["lo"], g.index["src"]] == pytest.approx(20.0)      # in region: scaled
    assert g.W_chem[g.index["hi"], g.index["src"]] == pytest.approx(-200.0)    # outside: factor 1.0
    p = g.params["normalize"]
    assert p["regions"] == ["central_brain"] and p["cells_scaled"] == 1 and p["cells_outside_regions"] == 2
    with pytest.raises(ValueError):
        graph.normalize_inputs(ds, ref, ds_types, ref_types, regions=["central_brain"])   # regions need cell_regions


def test_cache_name_carries_option_hash():
    plain = graph._cache_file("ds", "ref")
    opts = graph._cache_file("ds", "ref", {"strip_auto": True, "regions": ["central_brain"]})
    assert plain.name.startswith("ds__norm-ref-") and plain.suffix == ".npz"
    assert opts.name.startswith("ds__norm-ref-") and opts != plain
    assert graph._cache_file("ds", "ref", {"regions": ["central_brain"], "strip_auto": True}) == opts   # key order is irrelevant
    assert graph._cache_file("ds", "ref", {}) == plain and graph._cache_file("ds", "ref", {"strip_auto": False, "regions": None}) == plain


def test_normalize_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(paths, "DB_PATH", tmp_path / "nope.sqlite")     # no database: get goes through the cache only
    ds, ds_types, ref, ref_types = _pair()
    graph.save_cache(ds); graph.save_cache(ref)
    monkeypatch.setattr(graph, "_cell_types", lambda conn, dataset_id, names: ds_types if dataset_id == "ds" else ref_types)
    g = graph.get("ds", normalize="ref")
    assert graph._cache_file("ds", "ref").exists() and graph._cache_file("ds", "ref").parent == tmp_path / "cache"
    assert g.W_chem[g.index["lo"], g.index["src"]] == pytest.approx(20.0)
    g2 = graph.get("ds", normalize="ref")                               # second time — from cache
    assert g2.version == "v9 norm:ref" and g2.params["normalize"]["cells_scaled"] == 2
    assert (g2.W_chem != g.W_chem).nnz == 0
    assert graph.get("ds").version == "v9"                             # the ordinary cache is untouched
    monkeypatch.setattr(graph, "_cell_regions", lambda conn, dataset_id, names: ["central_brain", "central_brain", "optic_lobe", "", ""])
    g3 = graph.get("ds", normalize="ref", norm_opts={"strip_auto": True, "regions": ["central_brain"]})
    assert g3.params["normalize"]["regions"] == ["central_brain"] and g3.params["normalize"]["cells_scaled"] == 1
    assert graph._cache_file("ds", "ref", {"strip_auto": True, "regions": ["central_brain"]}).exists()
    assert graph.get("ds", normalize="ref").params["normalize"]["cells_scaled"] == 2    # the plain normalized cache is separate
