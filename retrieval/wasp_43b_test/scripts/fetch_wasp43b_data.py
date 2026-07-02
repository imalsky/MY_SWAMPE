#!/usr/bin/env python3
"""Fetch public WASP-43 b JWST/MIRI reduced products from Zenodo."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict
from urllib.request import urlopen

ZENODO_RECORD_ID = "10525170"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
ARCHIVE_NAME = "WASP43b_MIRI_Data.zip"

SUITE_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = SUITE_ROOT / "data" / "raw"
PROVENANCE_DIR = SUITE_ROOT / "data" / "provenance"


def fetch_json(url: str) -> Dict[str, Any]:
    """Fetch a JSON document from a URL."""
    with urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Compute a SHA-256 checksum for a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def zenodo_archive_url(record: Dict[str, Any]) -> str:
    """Return the content URL for the WASP-43 b archive."""
    for file_info in record.get("files", []):
        if file_info.get("key") == ARCHIVE_NAME:
            return str(file_info["links"]["self"])
    raise KeyError(f"{ARCHIVE_NAME} not found in Zenodo record {ZENODO_RECORD_ID}.")


def write_provenance(record: Dict[str, Any], archive_path: Path | None = None) -> None:
    """Write machine-readable provenance for the fetched data product."""
    PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "target": "WASP-43 b",
        "source": "Zenodo",
        "doi": "10.5281/zenodo.10525170",
        "record_api_url": ZENODO_API_URL,
        "archive_name": ARCHIVE_NAME,
        "archive_url": zenodo_archive_url(record),
        "metadata": {
            "title": record.get("metadata", {}).get("title"),
            "publication_date": record.get("metadata", {}).get("publication_date"),
            "creators": record.get("metadata", {}).get("creators", []),
        },
    }
    if archive_path is not None and archive_path.exists():
        payload["archive_path"] = str(archive_path)
        payload["archive_size_bytes"] = archive_path.stat().st_size
        payload["archive_sha256"] = sha256_file(archive_path)

    (PROVENANCE_DIR / "zenodo_10525170.json").write_text(json.dumps(payload, indent=2))


def download_archive(url: str, output_path: Path) -> None:
    """Download the Zenodo archive to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=120) as response, output_path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Write provenance metadata without downloading the archive.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download the archive even if it already exists.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RAW_DIR / ARCHIVE_NAME,
        help="Archive output path.",
    )
    return parser.parse_args()


def main() -> None:
    """Fetch metadata and optionally the archive.

    Offline-tolerant: when the archive is already on disk and the Zenodo API is
    unreachable (e.g. a cluster compute node without working egress), keep the
    existing archive + provenance and exit 0 instead of failing the job.
    """
    args = parse_args()
    try:
        record = fetch_json(ZENODO_API_URL)
    except Exception as exc:
        if args.output.exists() and not args.force:
            print(f"[offline: Zenodo API unreachable ({exc!r}); using existing {args.output}]")
            print(f"[sha256 {sha256_file(args.output)}]")
            return
        raise RuntimeError(
            f"Zenodo API unreachable and {args.output} does not exist. "
            "Run this script once on a machine with internet access (e.g. a login node)."
        ) from exc
    archive_url = zenodo_archive_url(record)

    if args.metadata_only:
        write_provenance(record, args.output if args.output.exists() else None)
        print(f"[wrote {PROVENANCE_DIR / 'zenodo_10525170.json'}]")
        return

    if args.output.exists() and not args.force:
        print(f"[using existing {args.output}]")
    else:
        print(f"[downloading {archive_url}]")
        download_archive(archive_url, args.output)

    checksum = sha256_file(args.output)
    write_provenance(record, args.output)
    print(f"[wrote {args.output}]")
    print(f"[sha256 {checksum}]")


if __name__ == "__main__":
    main()
