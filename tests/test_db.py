import json

from brainlab.store import db


def test_schema_and_roundtrip(tmp_db):
    conn = db.connect(tmp_db)
    db.register_dataset(conn, "toy", "unit-test", "1", "CC-BY", {"a.csv": "00"}, {"sign_rule": "test"})
    db.add_neurons(conn, "toy", [
        {"name": "A", "cell_type": "t", "cell_class": "neuron", "transmitter": "GABA", "side": "L", "region": "", "extra": {}},
        {"name": "B", "cell_type": "t", "cell_class": "neuron", "transmitter": "", "side": "R", "region": "", "extra": {"k": 1}},
    ])
    db.add_edges(conn, "toy", [
        {"pre": "A", "post": "B", "kind": "chemical", "count": 3, "sign": -1},
        {"pre": "A", "post": "B", "kind": "electrical", "count": 1, "sign": 0},
    ])
    info = db.dataset_info(conn, "toy")
    assert info["license"] == "CC-BY" and json.loads(info["params"])["sign_rule"] == "test"
    ns = db.neurons(conn, "toy")
    assert [n["name"] for n in ns] == ["A", "B"] and ns[1]["extra"] == {"k": 1}
    es = db.edges(conn, "toy")
    assert len(es) == 2 and es[0]["sign"] == -1 and es[1]["kind"] == "electrical"


def test_register_twice_replaces(tmp_db):
    conn = db.connect(tmp_db)
    db.register_dataset(conn, "toy", "s", "1", "L", {}, {})
    db.add_neurons(conn, "toy", [{"name": "A", "cell_type": "", "cell_class": "neuron", "transmitter": "", "side": "", "region": "", "extra": {}}])
    db.register_dataset(conn, "toy", "s", "2", "L", {}, {})
    assert db.neurons(conn, "toy") == []          # re-registering clears the dataset's old rows
    assert db.dataset_info(conn, "toy")["version"] == "2"


def test_rollback_on_failing_edge_generator(tmp_db):
    """register_dataset/add_neurons/add_edges do not commit themselves (task 5 of the final wave):
    if the edge generator fails midway and the loader calls conn.rollback(), the database keeps
    neither the dataset nor the neurons — no half-loaded state."""
    conn = db.connect(tmp_db)

    def bad_edges():
        yield {"pre": "A", "post": "B", "kind": "chemical", "count": 1, "sign": 1}
        raise RuntimeError("edge generator failed")

    try:
        db.register_dataset(conn, "toy", "s", "1", "L", {}, {})
        db.add_neurons(conn, "toy", [{"name": "A", "cell_type": "", "cell_class": "neuron", "transmitter": "", "side": "", "region": "", "extra": {}}])
        db.add_edges(conn, "toy", bad_edges())
        conn.commit()
    except RuntimeError:
        conn.rollback()
    assert db.dataset_info(conn, "toy") is None
    assert db.neurons(conn, "toy") == []
    conn.close()


def test_missing_dataset(tmp_db):
    conn = db.connect(tmp_db)
    assert db.dataset_info(conn, "nope") is None
    assert db.neurons(conn, "nope") == []
