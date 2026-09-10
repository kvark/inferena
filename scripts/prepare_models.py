#!/usr/bin/env python3
"""Fetch the declared immutable SmolLM2 inputs before offline collection."""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
PINS = json.loads((ROOT / "models/smollm2-revisions.json").read_text())


def prepare_model(name):
    directory = ROOT / "models" / name
    if directory.exists():
        raise ValueError(f"{directory} already exists; move it aside explicitly before preparing inputs")
    snapshot_download(f"HuggingFaceTB/{name}", revision=PINS[name],
                      allow_patterns=["config.json", "model.safetensors"], local_dir=directory)
    hashes = {}
    for filename in ("config.json", "model.safetensors"):
        with (directory / filename).open("rb") as source:
            hashes[filename] = hashlib.file_digest(source, "sha256").hexdigest()
    provenance = {"repo": f"HuggingFaceTB/{name}", "revision": PINS[name], "sha256": hashes}
    (directory / "source.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Prepared {name} at {PINS[name]}", flush=True)


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
