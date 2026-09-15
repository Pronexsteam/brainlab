import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from brainlab import paths  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: real data, slow")


@pytest.fixture
def tmp_db(tmp_path):
    """A separate database per test, so the working one is not touched."""
    return tmp_path / "store.sqlite"


@pytest.fixture(scope="session")
def worm_raw():
    d = paths.DATA / "worm"
    if not (d / "herm_full_edgelist.csv").exists():
        pytest.skip("no raw worm data in data/worm")
    return d
