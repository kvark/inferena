#!/usr/bin/env python3
"""Fetch the declared immutable SmolLM2 inputs before offline collection."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
PINS = json.loads((ROOT / "models/smollm2-revisions.json").read_text())
FILES = ("config.json", "model.safetensors")


def provenance(name, directory):
    pin = PINS[name]
    hashes = {}
    for filename in FILES:
        path = directory / filename
        if not path.is_file():
            raise ValueError(f"{directory} is incomplete: missing {filename}")
        with path.open("rb") as source:
            hashes[filename] = hashlib.file_digest(source, "sha256").hexdigest()
        if hashes[filename] != pin["sha256"][filename]:
            raise ValueError(f"{path} does not match the pinned checkpoint; move {directory} aside explicitly")
    return {"repo": f"HuggingFaceTB/{name}", "revision": pin["revision"], "sha256": hashes}


def prepare_model(name):
    directory = ROOT / "models" / name
    if directory.exists():
        receipt = provenance(name, directory)
        (directory / "source.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"Verified {name} at {receipt['revision']}", flush=True)
        return
    with tempfile.TemporaryDirectory(prefix=f".{name}.", dir=directory.parent) as temporary:
        staging = Path(temporary)
        snapshot_download(f"HuggingFaceTB/{name}", revision=PINS[name]["revision"],
                          allow_patterns=list(FILES), local_dir=staging)
        receipt = provenance(name, staging)
        (staging / "source.json").write_text(json.dumps(receipt, indent=2) + "\n")
        staging.replace(directory)
    print(f"Prepared {name} at {receipt['revision']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", choices=PINS)
    args = parser.parse_args()
    for name in args.models:
        try:
            prepare_model(name)
        except ValueError as error:
            parser.error(str(error))


if __name__ == "__main__":
    main()
