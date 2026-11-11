"""Atomic JSON replacement for single-writer local stores."""

import json
import math
import os
import tempfile
from pathlib import Path


def write_json(path, value):
    def sanitize(item):
        if isinstance(item, float) and not math.isfinite(item):
            return None
        if isinstance(item, dict):
            return {k: sanitize(v) for k, v in item.items()}
        if isinstance(item, list):
            return [sanitize(v) for v in item]
        return item

    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(sanitize(value), stream, indent=2, allow_nan=False, default=str)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
