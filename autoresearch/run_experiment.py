from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autoresearch.editable.recipe import hypothesis, what_to_try_next
from autoresearch.locked.anti_cheat_checks import locked_files_modified
from autoresearch.locked.evaluator import evaluate
from autoresearch.locked.scoring import ScoreResult, score_metrics


RUNS_DIR = ROOT / "autoresearch" / "runs"
LEADERBOARD_PATH = ROOT / "autoresearch" / "leaderboard.csv"
JOURNAL_PATH = ROOT / "autoresearch" / "research_journal.md"


def _run_command(args: list[str]) -> tuple[bool, str]:
    completed = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return completed.returncode == 0, completed.stdout


def _latest_accepted_metrics() -> dict[str, float] | None:
    if not LEADERBOARD_PATH.exists():
        return None
    with LEADERBOARD_PATH.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("accepted") == "true"]
    if not rows:
        return None
    best = max(rows, key=lambda row: float(row.get("score", "0") or 0.0))
    return {
        "crash_rate": float(best.get("crash_rate", "0") or 0.0),
        "flythrough_success_rate": float(best.get("flythrough_success_rate", "0") or 0.0),
    }


def _git_diff() -> str:
    completed = subprocess.run(
        ["git", "diff", "--", "autoresearch/editable"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return completed.stdout or "No tracked editable diff for this run.\n"


def _changed_editable_files() -> str:
    completed = subprocess.run(
        ["git", "status", "--short", "--", "autoresearch/editable"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    changed = [line[3:] for line in completed.stdout.splitlines() if line.strip()]
    return ", ".join(changed) if changed else "none"


def _write_proposal(run_dir: Path, mode: str) -> None:
    run_dir.joinpath("proposal.md").write_text(
        f"# Proposal\n\nMode: `{mode}`\n\nHypothesis: {hypothesis()}\n",
        encoding="utf-8",
    )


def _write_eval_summary(run_dir: Path, result: dict, score: ScoreResult) -> None:
    metrics = result["metrics"]
    reasons = ", ".join(score.rejection_reasons) if score.rejection_reasons else "none"
    run_dir.joinpath("eval_summary.md").write_text(
        "\n".join(
            [
                "# Evaluation Summary",
                "",
                f"Mode: `{result['mode']}`",
                f"Accepted: `{str(score.accepted).lower()}`",
                f"Score: `{score.score:.6f}`",
                f"Rejection reasons: {reasons}",
                "",
                "## Metrics",
                "",
                *[f"- `{key}`: `{value}`" for key, value in sorted(metrics.items())],
                "",
            ]
        ),
        encoding="utf-8",
    )


def _append_leaderboard(run_id: str, mode: str, metrics: dict[str, float], score: ScoreResult) -> None:
    fieldnames = [
        "run_id",
        "timestamp_utc",
        "mode",
        "accepted",
        "score",
        "flythrough_success_rate",
        "first_acquisition_rate",
        "mean_steps_to_acquire",
        "mean_steps_to_intercept",
        "crash_rate",
        "out_of_bounds_rate",
        "lost_target_rate",
        "side_pass_false_success_rate",
        "rejection_reasons",
    ]
    exists = LEADERBOARD_PATH.exists()
    with LEADERBOARD_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "run_id": run_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "mode": mode,
                "accepted": str(score.accepted).lower(),
                "score": f"{score.score:.6f}",
                "rejection_reasons": "; ".join(score.rejection_reasons),
                **{key: metrics.get(key, 0.0) for key in fieldnames if key in metrics},
            }
        )


def _append_journal(
    run_id: str,
    mode: str,
    metrics: dict[str, float],
    score: ScoreResult,
    changed_files: str,
) -> None:
    result = "accepted" if score.accepted else "rejected"
    JOURNAL_PATH.open("a", encoding="utf-8").write(
        "\n".join(
            [
                f"\n## {run_id}",
                "",
                f"- Hypothesis: {hypothesis()}",
                f"- Mode: `{mode}`",
                f"- Files changed: {changed_files}",
                f"- Result: {result}",
                f"- Score: `{score.score:.6f}`",
                f"- Accepted or rejected: {result}",
                f"- Rejection reasons: {', '.join(score.rejection_reasons) if score.rejection_reasons else 'none'}",
                f"- What to try next: {what_to_try_next(metrics)}",
                "",
            ]
        ),
    )


def run_experiment(mode: str) -> int:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    _write_proposal(run_dir, mode)
    check_env_passed, check_env_output = _run_command([sys.executable, "scripts/check_env.py"])
    tests_passed, tests_output = _run_command([sys.executable, "-m", "pytest", "-q"])
    result = evaluate(mode=mode)
    metrics = result["metrics"]
    run_dir.joinpath("metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    run_dir.joinpath("diff.patch").write_text(_git_diff(), encoding="utf-8")
    run_dir.joinpath("validation.log").write_text(
        "# check_env\n\n" + check_env_output + "\n# pytest\n\n" + tests_output,
        encoding="utf-8",
    )

    score = score_metrics(
        metrics,
        tests_passed=tests_passed,
        check_env_passed=check_env_passed,
        locked_files_modified=locked_files_modified(),
        baseline_metrics=_latest_accepted_metrics(),
    )
    _write_eval_summary(run_dir, result, score)
    _append_leaderboard(run_id, mode, metrics, score)
    _append_journal(run_id, mode, metrics, score, _changed_editable_files())

    print(f"run_id={run_id}")
    print(f"run_dir={run_dir}")
    print(f"accepted={str(score.accepted).lower()}")
    print(f"score={score.score:.6f}")
    if score.rejection_reasons:
        print("rejection_reasons=" + "; ".join(score.rejection_reasons))
    return 0 if score.accepted else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "medium"], required=True)
    args = parser.parse_args()
    raise SystemExit(run_experiment(args.mode))


if __name__ == "__main__":
    main()

