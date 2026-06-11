# Policy — Mandatory Anti-Hardcoding Audit

> Reference for the STAI-X Award B workflow. Extracted from CLAUDE.md so the dispatcher stays lean.
> Read by `hardcoding-and-feature-auditor`. Pairs with `prohibited_assumptions.md`.

The `hardcoding-and-feature-auditor` agent runs in multiple modes during the workflow:

| When | Mode | Trigger |
|------|------|---------|
| Step 5, each round | `feature-audit` | After programming + model search |
| Step 7 (if supervisor flags it) | `post` | Before finalising outputs |

## Consumed-artifact rule (feature-audit / leakage)

Audit the feature set the model **actually trains on**, resolved dynamically by reading what the
Phase-B modeling entry point consumes — not whatever matrix is on disk. A written feature artifact
that no training entry point reads is **unconsumed**: report it `WARN`, never a model-leakage `FAIL`.
Leakage is decided by *where* a target aggregate is computed: an in-pipeline transformer fit per fold
is SAFE; a value precomputed over the full training data is CV-leaky. Every leakage finding cites the
consumed file and confirms it is read.

## What it searches for

**Static suspicious terms** (always searched):
- Award A field names: `rate_per_10000_ed_visits`, `overdose_category`, `all_drugs`, `all_opioids`, `all_stimulants`
- Known competition-specific column names: `survived`, `passengerid`, `casual`, `registered`, `saleprice`, etc.
- Assumed file names: `train.csv`, `test.csv`, `sampleSubmission.csv`, etc.
- Domain vocabulary: `overdose`, `opioid`, `stimulant`, `titanic`, etc.
- Magic numbers: `918`

**Dynamic suspicious terms** (extracted from the current dataset at runtime):
- Backtick-quoted identifiers in `DATA_DESCRIPTION.md`
- Column names from the training file header
- Column names from the sample submission header

## Classification rules

| Classification | Condition |
|---|---|
| **acceptable** | In `tests/`, comments (`#`), docstrings, or markdown prose; OR term appears in a list of 3+ candidate fallbacks |
| **risky** | Term in a 1–2 item list, `in`-operator check, or `!=` comparison |
| **unacceptable** | Direct column access `df["term"]`, variable assignment `var = "term"`, equality check `== "term"`, file loading with hardcoded name |

A `fail` verdict is logged and surfaced but does **not** halt the workflow.

## Manual execution

```bash
python scripts/audit_hardcoding.py           # pre-run
python scripts/audit_hardcoding.py --verbose
python scripts/audit_hardcoding.py --phase post
```
