"""База схем: наборы, нейроны, связи. Одна SQLite на всё, набор — столбец dataset.

Перерегистрация набора удаляет его старые нейроны и связи: база всегда строится
из сырых файлов заново, а не правится по месту.
"""
import hashlib
import json
import sqlite3
from pathlib import Path

from .. import paths

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY, source TEXT, version TEXT, license TEXT,
    files TEXT, params TEXT, loaded_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS neurons (
    dataset TEXT, name TEXT, cell_type TEXT, cell_class TEXT, transmitter TEXT,
    side TEXT, region TEXT, extra TEXT,
    PRIMARY KEY (dataset, name));
CREATE TABLE IF NOT EXISTS edges (
    dataset TEXT, pre TEXT, post TEXT, kind TEXT, count REAL, sign INTEGER);
CREATE INDEX IF NOT EXISTS edges_ds ON edges (dataset);
"""


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def connect(path=None):
    path = Path(path) if path else paths.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def register_dataset(conn, dataset_id, source, version, license, files, params):
    """Не коммитит: набор регистрируется одной транзакцией с add_neurons/add_edges на уровне
    загрузчика (commit там же, rollback при исключении — задача 5 финальной волны), иначе
    упавший на середине генератор рёбер оставлял бы в базе набор без части/всех рёбер."""
    conn.execute("DELETE FROM neurons WHERE dataset = ?", (dataset_id,))
    conn.execute("DELETE FROM edges WHERE dataset = ?", (dataset_id,))
    conn.execute("INSERT OR REPLACE INTO datasets (id, source, version, license, files, params) VALUES (?,?,?,?,?,?)",
                 (dataset_id, source, version, license, json.dumps(files, ensure_ascii=False), json.dumps(params, ensure_ascii=False)))


def add_neurons(conn, dataset_id, rows):
    """Не коммитит — см. register_dataset."""
    conn.executemany(
        "INSERT OR REPLACE INTO neurons VALUES (?,?,?,?,?,?,?,?)",
        [(dataset_id, r["name"], r.get("cell_type", ""), r.get("cell_class", ""), r.get("transmitter", ""),
          r.get("side", ""), r.get("region", ""), json.dumps(r.get("extra", {}), ensure_ascii=False)) for r in rows])


def add_edges(conn, dataset_id, rows, chunk=200_000):
    """Вставка порциями (на 3–15 млн рёбер список кортежей целиком не нужен в памяти), но не
    коммитит — см. register_dataset."""
    buf = []
    for r in rows:
        buf.append((dataset_id, r["pre"], r["post"], r["kind"], float(r["count"]), int(r.get("sign", 0))))
        if len(buf) >= chunk:
            conn.executemany("INSERT INTO edges VALUES (?,?,?,?,?,?)", buf)
            buf = []
    if buf:
        conn.executemany("INSERT INTO edges VALUES (?,?,?,?,?,?)", buf)


def dataset_info(conn, dataset_id):
    row = conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
    return dict(row) if row else None


def neurons(conn, dataset_id):
    out = []
    for row in conn.execute("SELECT * FROM neurons WHERE dataset = ? ORDER BY name", (dataset_id,)):
        d = dict(row)
        d["extra"] = json.loads(d["extra"] or "{}")
        out.append(d)
    return out


def edges(conn, dataset_id):
    return [dict(r) for r in conn.execute("SELECT * FROM edges WHERE dataset = ? ORDER BY rowid", (dataset_id,))]
