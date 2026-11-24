"""Dataset download: streamed gzip -> CSV, size check, partial files never left behind."""

import gzip
import importlib.util

import httpx
import pytest

from src.config import PROJECT_ROOT

spec = importlib.util.spec_from_file_location("dl", PROJECT_ROOT / "scripts" / "00_download_data.py")
dl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dl)

CSV = b"event_time,event_type\n" + b"".join(b"2019-10-01 00:00:00 UTC,view\n" for _ in range(50_000))
GZ = gzip.compress(CSV)


def client(body: bytes) -> httpx.Client:
    def handler(request):
        return httpx.Response(200, content=body, headers={"content-length": str(len(body))})
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_url_points_at_public_gzip():
    assert dl.url_for("2019-Oct.csv") == "https://data.rees46.com/datasets/marketplace/2019-Oct.csv.gz"


def test_download_decompresses(tmp_path):
    c = client(GZ)
    dest = tmp_path / "2019-Oct.csv"
    dl.download(c, "https://x/2019-Oct.csv.gz", dest, dl.remote_size(c, "https://x/2019-Oct.csv.gz"))
    assert dest.read_bytes() == CSV and not list(tmp_path.glob("*.part"))


def test_truncated_download_is_discarded(tmp_path):
    dest = tmp_path / "2019-Oct.csv"
    with pytest.raises(RuntimeError):
        dl.download(client(GZ[:-1000]), "https://x/f.gz", dest, len(GZ))
    assert not dest.exists() and not list(tmp_path.glob("*.part"))
