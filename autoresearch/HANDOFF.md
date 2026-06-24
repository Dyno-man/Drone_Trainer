# Autoresearch Handoff

Goal: improve viewport-based search and true fly-through interception without rebuilding the environment.

Primary metric: `flythrough_success_rate`

Hard rejection gates:
- `side_pass_false_success_rate > 0`
- `scripts/check_env.py` fails
- `pytest -q` fails
- locked evaluator or benchmark files are modified
- `crash_rate` gets significantly worse than the best accepted run

## Edit Only These Files

- `autoresearch/editable/recipe.py`
- `autoresearch/editable/reward_weights.py`
- `autoresearch/editable/curriculum.py`
- `autoresearch/editable/policy_config.py`

Do not edit locked files during research trials:

- `autoresearch/locked/evaluator.py`
- `autoresearch/locked/scoring.py`
- `autoresearch/locked/benchmark_scenarios.py`
- `autoresearch/locked/anti_cheat_checks.py`
- `autoresearch/locked_manifest.json`

## Standard Trial Loop

1. Inspect the leaderboard:

```bash
python autoresearch/compare_runs.py
```

2. Form one small hypothesis.

3. Modify only files in `autoresearch/editable/`.

4. Run a quick trial:

```bash
python autoresearch/run_experiment.py --mode quick
```

5. If accepted and promising, run a medium trial:

```bash
python autoresearch/run_experiment.py --mode medium
```

6. Compare results:

```bash
python autoresearch/compare_runs.py
```

7. Keep accepted changes only if they improve the objective and do not trip any rejection gate.

## Branch-Based Trial

Use this when starting a new isolated research attempt from the stable baseline branch.

```bash
python autoresearch/run_experiment.py --mode quick --create-branch
```

Default behavior:
- base branch: `Auto-Research`
- new branch: `codex/autoresearch-<run_id>`

Optional explicit branch:

```bash
python autoresearch/run_experiment.py --mode quick --create-branch --branch-name codex/my-research-idea
```

The worktree must be clean before using `--create-branch`.

## Evaluate A Saved Model

```bash
python autoresearch/locked/evaluator.py --model path/to/model.zip
```

Use this for model inspection only. Research changes should still go through `run_experiment.py`.

## Generated Outputs

Each experiment writes an ignored run folder:

```text
autoresearch/runs/<run_id>/
```

Expected files:
- `proposal.md`
- `metrics.json`
- `diff.patch`
- `eval_summary.md`
- `validation.log`

Persistent tracked summaries:
- `autoresearch/leaderboard.csv`
- `autoresearch/research_journal.md`

## Decision Rules

Accept:
- run is marked `accepted=true`
- `flythrough_success_rate` improves or supports the current research direction
- `side_pass_false_success_rate` remains `0.0`
- crash and out-of-bounds rates do not regress materially

Reject:
- any hard gate trips
- improvement is only proximity-like behavior rather than fly-through interception
- changes require touching locked files
- results are noisy without a clear hypothesis

## Minimal Fresh-Context Prompt

```text
Use autoresearch/HANDOFF.md. Improve viewport search and fly-through interception.
Only edit autoresearch/editable/*.py. Run quick experiments, compare leaderboard,
and keep only accepted changes that improve flythrough_success_rate without side-pass false success.
```
