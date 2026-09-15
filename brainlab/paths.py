"""Пути проекта. Всё считается от корня, абсолютных путей в модулях нет."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = DATA / "cache"
RESULTS = ROOT / "results"
LAB = ROOT / "lab"
DOCS = ROOT / "docs"
DB_PATH = DATA / "store.sqlite"


def ensure():
    for p in (DATA, CACHE, RESULTS, LAB / "experiments", DOCS):
        p.mkdir(parents=True, exist_ok=True)
