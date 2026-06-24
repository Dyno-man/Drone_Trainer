from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUTORESEARCH_DIR = ROOT / "autoresearch"
MANIFEST_PATH = AUTORESEARCH_DIR / "locked_manifest.json"
LOCKED_RELATIVE_PATHS = (
    "autoresearch/locked/evaluator.py",
    "autoresearch/locked/scoring.py",
    "autoresearch/locked/benchmark_scenarios.py",
    "autoresearch/locked/anti_cheat_checks.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def current_locked_hashes() -> dict[str, str]:
    return {relative: _sha256(ROOT / relative) for relative in LOCKED_RELATIVE_PATHS}


def write_manifest() -> None:
    MANIFEST_PATH.write_text(
        json.dumps(current_locked_hashes(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def locked_files_modified() -> bool:
    if not MANIFEST_PATH.exists():
        return True
    expected = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return current_locked_hashes() != expected


def assert_locked_files_clean() -> None:
    if locked_files_modified():
        raise RuntimeError("Locked autoresearch files differ from locked_manifest.json")


if __name__ == "__main__":
    assert_locked_files_clean()
    print("Locked autoresearch files match manifest.")

