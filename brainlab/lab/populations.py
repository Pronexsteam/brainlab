"""A dataset's cell groups, by the rules in lab/populations/<dataset>.yaml (design §7a).
The rules read columns of the neurons table and extra keys; name lists are cached in
data/cache/<dataset>_populations.json and recomputed if the yaml is newer than the cache."""
import json
import os

import yaml

from .. import paths
from ..store import db


def _file(dataset_id):
    return paths.LAB / "populations" / ("%s.yaml" % dataset_id)


def load(dataset_id):
    f = _file(dataset_id)
    if not f.exists():
        raise KeyError("no groups file %s" % f)
    return yaml.safe_load(f.read_text(encoding="utf-8"))


def _match_value(actual, cond):
    actual = "" if actual is None else str(actual)
    if isinstance(cond, dict):
        if "startswith" in cond: return actual.startswith(str(cond["startswith"]))
        if "contains" in cond: return str(cond["contains"]) in actual
        if "in" in cond: return actual in [str(x) for x in cond["in"]]
        raise ValueError("unrecognized condition %r" % cond)
    return actual == str(cond)


def _match(row, where):
    for field, cond in where.items():
        if field == "any":
            if not any(_match(row, w) for w in cond):
                return False
            continue
        # a row's column always takes precedence over extra, even if it is "" (empty in the column
        # is still an answer, not a reason to look in extra); we look in extra only for fields outside the table columns.
        actual = row.get(field, row["extra"].get(field)) if field in row else row["extra"].get(field)
        if not _match_value(actual, cond):
            return False
    return True


def _current_version(dataset_id, conn):
    """The dataset's version in the database right now (for comparison with the version recorded in the groups cache)."""
    owns = conn is None
    conn = conn or db.connect()
    try:
        return (db.dataset_info(conn, dataset_id) or {}).get("version", "")
    finally:
        if owns:
            conn.close()


def resolve(dataset_id, conn=None):
    spec = load(dataset_id)
    cache = paths.CACHE / ("%s_populations.json" % dataset_id)
    ymtime = os.path.getmtime(_file(dataset_id))
    if cache.exists():
        try:
            c = json.loads(cache.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            c = {}
        # the cache is valid if the yaml has not changed AND (there is no db / the db is not newer than the cache OR the
        # dataset version in the cache matches the current one — changing data in the db changes version, task 2 of the final wave)
        db_fresh = True
        if paths.DB_PATH.exists():
            db_mtime = os.path.getmtime(paths.DB_PATH)
            cache_mtime = os.path.getmtime(cache)
            db_fresh = db_mtime <= cache_mtime or c.get("version") == _current_version(dataset_id, conn)
        if c.get("yaml_mtime") == ymtime and "groups" in c and db_fresh:
            return c["groups"]
    owns = conn is None
    conn = conn or db.connect()
    try:
        rows = db.neurons(conn, dataset_id)
        version = (db.dataset_info(conn, dataset_id) or {}).get("version", "")
    finally:
        if owns:
            conn.close()
    known = {r["name"] for r in rows}
    out = {}
    for name, g in spec["groups"].items():
        if "names" in g:
            out[name] = [str(x) for x in g["names"] if str(x) in known]
        else:
            out[name] = [r["name"] for r in rows if _match(r, g["where"])]
    paths.CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"yaml_mtime": ymtime, "version": version, "groups": out}, ensure_ascii=False), encoding="utf-8")
    return out


def names(dataset_id, group, conn=None):
    r = resolve(dataset_id, conn)
    if group not in r:
        raise KeyError("dataset %s has no group %s (available: %s)" % (dataset_id, group, sorted(r)))
    return r[group]


def state_groups(dataset_id):
    return list(load(dataset_id).get("state", []))


def report(dataset_id):
    r = resolve(dataset_id)
    return {k: len(v) for k, v in r.items()}
