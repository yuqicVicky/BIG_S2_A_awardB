# Policy — Prohibited Assumptions

> Reference for the STAI-X Award B workflow. Extracted from CLAUDE.md so the dispatcher stays lean.
> Every agent that references a column, file, metric, or count must obey this. Enforced by
> `hardcoding-and-feature-auditor` (see `hardcoding_audit.md`).

Do not hardcode:

- `rate_per_10000_ed_visits`
- `overdose_category`
- `all_drugs`, `all_opioids`, or `all_stimulants`
- `918` rows
- any fixed period-id map
- any local absolute path from a developer machine
- any column name, file name, or domain term not derived from `DATA_DESCRIPTION.md` or the data files at runtime

All column names, file paths, task types, and metric names must be resolved dynamically by the agents
from the actual data and description files. Every agent that references a column name must read it from
`spec_parse.json` or `data_profile.json` — never hardcode it.

This extends to **runtime budgets**: no fixed seed counts, tuning iterations, fold counts, or
second-budgets baked into any agent. Budgets are derived from the wall-clock that remains (see
`modeling_rules.md` and the `modeling-watchdog` agent).
