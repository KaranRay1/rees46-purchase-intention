"""Step 0: download the REES46 dataset from its public source (no account or API key needed).

The data is not shipped with the project (about 53 GB as CSV). This script fetches the monthly
files from REES46's public server, decompresses them on the fly and writes them where step 1
expects them (``data/raw/<month>.csv``, see ``config.yaml -> paths.raw_dir``).

Source: REES46 "eCommerce behavior data from a multi-category store"
  * dataset page: https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store
    (Oct and Nov 2019 on Kaggle; the page links to the other months)
  * direct files: https://data.rees46.com/datasets/marketplace/<month>.csv.gz   (used here)

Sizes (compressed download -> CSV on disk): about 1.7 -> 5.7 GB for October 2019 alone, and
about 17.7 -> 53 GB for all seven months. Files already present are skipped, so the script can
be re-run after an interruption.

Examples::

    python scripts/00_download_data.py --mode dev      # 2019-Oct only (enough for --mode dev)
    python scripts/00_download_data.py --mode full     # all seven months (Oct 2019 - Apr 2020)
    python scripts/00_download_data.py --check         # only check that the links work
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import zlib
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config, resolve_path  # noqa: E402

BASE_URL = "https://data.rees46.com/datasets/marketplace/{name}.gz"
DATASET_PAGE = "https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store"
CSV_PER_GZ = 3.0          # a CSV is roughly 3x its gzip size (for the disk-space check)


def url_for(csv_name: str) -> str:
    return BASE_URL.format(name=csv_name)


def remote_size(client: httpx.Client, url: str) -> int | None:
    r = client.head(url, follow_redirects=True)
    if r.status_code != 200:
        return None
    return int(r.headers.get("content-length", 0)) or None


def download(client: httpx.Client, url: str, dest: Path, size: int | None) -> None:
    """Stream ``url`` (gzip) into ``dest`` (plain CSV), via a ``.part`` file renamed at the end."""
    part = dest.with_suffix(dest.suffix + ".part")
    gunzip = zlib.decompressobj(16 + zlib.MAX_WBITS)
    done, t0, last = 0, time.time(), 0.0
    with client.stream("GET", url, follow_redirects=True) as r, part.open("wb") as out:
        r.raise_for_status()
        for chunk in r.iter_bytes(1 << 20):
            out.write(gunzip.decompress(chunk))
            done += len(chunk)
            now = time.time()
            if now - last > 2 or (size and done >= size):
                last = now
                rate = done / max(now - t0, 1e-6) / 1e6
                pct = f"{done / size:6.1%}" if size else ""
                print(f"\r   {dest.name}: {done / 1e9:5.2f} GB {pct}  {rate:5.1f} MB/s", end="", flush=True)
        out.write(gunzip.flush())
    print()
    if size and done != size:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"{dest.name}: got {done:,} of {size:,} bytes; run the script again")
    part.rename(dest)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=["dev", "full"], default="dev",
                   help="dev = 2019-Oct only; full = all seven months (default: dev)")
    p.add_argument("--months", nargs="+", metavar="NAME", help="explicit files, e.g. 2019-Oct.csv 2019-Nov.csv")
    p.add_argument("--check", action="store_true", help="only check that the links work and print the sizes")
    args = p.parse_args(argv)

    cfg = load_config()
    names = args.months or cfg["data"][f"{args.mode}_files"]
    raw = Path(resolve_path(cfg, "raw_dir"))
    raw.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=httpx.Timeout(60.0, read=300.0), headers={"User-Agent": "rees46-coursework-download"}) as c:
        sizes = {n: remote_size(c, url_for(n)) for n in names}
        missing = [n for n, s in sizes.items() if s is None]
        for n, s in sizes.items():
            have = (raw / n).exists()
            print(f"{n:<14} {'%.2f GB' % (s / 1e9) if s else 'NOT FOUND':>10}  {'(already in data/raw)' if have else ''}")
        if missing:
            raise SystemExit(f"Not available at {BASE_URL.format(name='...')}: {missing}. Download them from the "
                             f"dataset page instead and put the CSVs in {raw}: {DATASET_PAGE}")
        if args.check:
            return
        todo = [n for n in names if not (raw / n).exists()]
        need = sum(sizes[n] for n in todo) * CSV_PER_GZ
        free = shutil.disk_usage(raw).free
        print(f"to download: {len(todo)} file(s), ~{sum(sizes[n] for n in todo) / 1e9:.1f} GB download, "
              f"~{need / 1e9:.0f} GB on disk; free: {free / 1e9:.0f} GB")
        if need > free:
            raise SystemExit("Not enough free disk space; free some space or use --months to fetch fewer files.")
        for n in todo:
            download(c, url_for(n), raw / n, sizes[n])
    print(f"Done. Next: python scripts/01_prepare_data.py --mode {args.mode}"
          + (" --set data.session_sample_fraction=0.1" if args.mode == "full" else ""))


if __name__ == "__main__":
    main()
