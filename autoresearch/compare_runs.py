from __future__ import annotations

import csv
import fcntl
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


ROOT = Path(__file__).resolve().parents[1]
LEADERBOARD_PATH = ROOT / "autoresearch" / "leaderboard.csv"
RESULTS_LOCK_PATH = ROOT / "autoresearch" / "results.lock"


@contextmanager
def _shared_results_lock() -> Iterator[None]:
    RESULTS_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_LOCK_PATH.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def main() -> None:
    if not LEADERBOARD_PATH.exists():
        print("No leaderboard exists yet. Run python autoresearch/run_experiment.py --mode quick first.")
        return

    with _shared_results_lock():
        with LEADERBOARD_PATH.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    if not rows:
        print("Leaderboard is empty.")
        return

    latest = rows[-1]
    accepted = [row for row in rows if row.get("accepted") == "true"]
    best = max(accepted or rows, key=lambda row: float(row.get("score", "0") or 0.0))
    latest_score = float(latest.get("score", "0") or 0.0)
    best_score = float(best.get("score", "0") or 0.0)
    delta = latest_score - best_score

    print(f"latest_run={latest['run_id']} accepted={latest['accepted']} score={latest_score:.6f}")
    print(f"best_run={best['run_id']} accepted={best['accepted']} score={best_score:.6f}")
    print(f"delta_vs_best={delta:.6f}")
    print(f"flythrough_success_rate={latest.get('flythrough_success_rate', '0')}")
    print(f"side_pass_false_success_rate={latest.get('side_pass_false_success_rate', '0')}")
    if latest.get("rejection_reasons"):
        print(f"rejection_reasons={latest['rejection_reasons']}")


if __name__ == "__main__":
    main()
