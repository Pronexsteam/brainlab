from pathlib import Path

from brainlab import paths
from brainlab.store import fetch


def test_fetch_skips_existing_and_downloads_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    monkeypatch.setattr(fetch, "FILES", {"toy": [("http://x/a.bin", "a.bin"), ("http://x/b.bin", "b.bin")]})
    raw = tmp_path / "toy" / "raw"
    raw.mkdir(parents=True)
    (raw / "a.bin").write_bytes(b"already")
    calls = []

    def fake(url, dest):
        calls.append(url)
        Path(dest).write_bytes(b"new")

    got = fetch.fetch("toy", download=fake)
    assert calls == ["http://x/b.bin"]
    assert sorted(p.name for p in got) == ["a.bin", "b.bin"]
    assert (raw / "a.bin").read_bytes() == b"already"
    m = fetch.manifest("toy")
    assert set(m) == {"a.bin", "b.bin"} and len(m["a.bin"]) == 64


def test_fetch_unknown_dataset():
    import pytest
    with pytest.raises(KeyError):
        fetch.fetch("no_such_dataset", download=lambda u, d: None)
