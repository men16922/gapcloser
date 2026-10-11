"""Environment settings: TETHER_<NAME>, falling back to the pre-rename GAPCLOSER_<NAME> so existing setups keep working."""

from __future__ import annotations

import os


def setting(name: str, default: str | None = None) -> str | None:
    return os.environ.get(f"TETHER_{name}", os.environ.get(f"GAPCLOSER_{name}", default))
