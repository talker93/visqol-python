"""Fetch only the official conformance files, pinned by commit and Git blob SHA."""

from __future__ import annotations

import argparse
import hashlib
import json
import ssl
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen


def blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".testdata"))
    args = parser.parse_args()
    manifest = json.loads(Path(__file__).with_name("conformance_files.json").read_text())
    base = f"https://raw.githubusercontent.com/google/visqol/{manifest['commit']}/testdata/"
    context = ssl.create_default_context()
    # Some python.org macOS installations lack the optional CA setup step.
    # Add certifi's verified roots when available; never disable verification.
    try:
        import certifi
    except ImportError:
        pass
    else:
        context.load_verify_locations(cafile=certifi.where())

    def fetch(entry: dict) -> None:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid manifest path: {relative}")
        target = args.output / relative
        if target.exists() and blob_sha(target.read_bytes()) == entry["sha"]:
            return
        with urlopen(base + entry["path"], timeout=60, context=context) as response:
            data = response.read(entry["size"] + 1)
        if len(data) != entry["size"] or blob_sha(data) != entry["sha"]:
            raise ValueError(f"Size or checksum mismatch: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".partial")
        temporary.write_bytes(data)
        temporary.replace(target)
        print(f"Verified {relative}", flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, manifest["files"]))
    print(f"Verified {len(manifest['files'])} files in {args.output}")


if __name__ == "__main__":
    main()
