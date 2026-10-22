"""Validate checkpoint release manifests without deserializing model weights."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(path: Path, *, allow_missing: bool = False) -> dict[str, Any]:
    """Check manifest structure, registered files and any declared weight hashes.

    Architecture and state-dictionary compatibility are checked by the export
    adapters when loading each checkpoint, rather than inferred from filenames.
    """
    path = path.expanduser().resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("models"), list):
        raise ValueError("Checkpoint manifest requires schema_version=1 and a models array")
    from .metrics.registry import TRANS

    allowed_models = {"bloom-unidirectional", "bloom-bidirectional", *TRANS}
    entries, identities = [], set()
    for index, row in enumerate(manifest["models"]):
        if not isinstance(row, dict):
            raise ValueError(f"Manifest entry {index} must be an object")
        required = ("model", "dataset", "panel", "direction", "checkpoint")
        missing = [k for k in required if not isinstance(row.get(k), str) or not row[k].strip()]
        if missing:
            raise ValueError(f"Manifest entry {index} is missing: {', '.join(missing)}")
        if row["model"] not in allowed_models:
            raise ValueError(f"Entry {index}: unsupported model {row['model']!r}")
        if row["dataset"] not in {"prenatal", "cortex", "sted"}:
            raise ValueError(f"Entry {index}: dataset must be prenatal, cortex or sted")
        if row["panel"] not in {"author", "hvg"}:
            raise ValueError(f"Entry {index}: panel must be author or hvg")
        if row["direction"] not in {"forward", "reverse", "bidirectional"}:
            raise ValueError(f"Entry {index}: invalid direction")
        if row["model"] == "bloom-bidirectional" and row["direction"] != "bidirectional":
            raise ValueError(f"Entry {index}: bidirectional BLOOM uses direction=bidirectional")
        if row["model"] == "bloom-unidirectional" and row["direction"] == "bidirectional":
            raise ValueError(
                f"Entry {index}: unidirectional BLOOM requires a forward or reverse entry"
            )
        identity = tuple(row[k] for k in ("model", "dataset", "panel", "direction"))
        if identity in identities:
            raise ValueError(f"Duplicate checkpoint slot: {identity}")
        identities.add(identity)
        files = {}
        errors = []
        for key in ("checkpoint", "metadata", "training_log"):
            if not row.get(key):
                continue
            candidate = Path(row[key]).expanduser()
            candidate = (
                (path.parent / candidate).resolve()
                if not candidate.is_absolute()
                else candidate.resolve()
            )
            exists = candidate.is_file()
            files[key] = {"path": str(candidate), "exists": exists}
            if not exists:
                errors.append(f"Missing {key}")
            elif key == "metadata":
                try:
                    data = json.loads(candidate.read_text())
                    if not isinstance(data, dict):
                        errors.append("Metadata must contain a JSON object")
                except (json.JSONDecodeError, UnicodeDecodeError):
                    errors.append("Metadata is not valid UTF-8 JSON")
        if row.get("sha256"):
            if not re.fullmatch("[a-fA-F0-9]{64}", row["sha256"]):
                errors.append("sha256 must be a real 64-character hexadecimal digest")
            elif files["checkpoint"]["exists"]:
                observed = _sha256(Path(files["checkpoint"]["path"]))
                files["checkpoint"]["sha256"] = observed
                if observed.lower() != row["sha256"].lower():
                    errors.append("Checkpoint SHA-256 mismatch")
        only_missing = bool(errors) and all(e.startswith("Missing ") for e in errors)
        status = "pending" if only_missing and allow_missing else "invalid" if errors else "ready"
        entries.append(
            {
                "index": index,
                **{k: row[k] for k in required[:-1]},
                "status": status,
                "files": files,
                "issues": errors,
            }
        )
    return {
        "manifest": str(path),
        "schema_version": 1,
        "valid": all(e["status"] != "invalid" for e in entries),
        "registered_models": len(entries),
        "ready": sum(e["status"] == "ready" for e in entries),
        "pending": sum(e["status"] == "pending" for e in entries),
        "entries": entries,
        "validation_scope": "Manifest structure, file presence, JSON metadata and optional exact weight hashes; no weight deserialization or architecture inference.",
    }
