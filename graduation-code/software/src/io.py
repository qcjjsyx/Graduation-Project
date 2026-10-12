from __future__ import annotations

import json


def write_manifest(path: str, manifest: dict) -> None:
    """Write a strict-JSON manifest (NaN/Inf are rejected)."""
    with open(path, "w", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
