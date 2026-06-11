# Policy — Datetime Feature Engineering: Invariant Behaviour

> Reference for the STAI-X Award B workflow. Extracted from CLAUDE.md so the dispatcher stays lean.
> Read by `analysis-programmer`, `analysis-planner`, and `hardcoding-and-feature-auditor`.

This rule applies to every unknown future dataset, not only the current one.

## What the agent MUST do

1. **Scan every column for datetime parseability**, including the row_id column and join keys.
2. **Never add the raw row_id / join key to the model feature set** — only derived `col__<field>` columns.
3. **Extract the full set of generic time features** from every detected datetime source column:
   - Always: `year`, `month`, `month_sin`, `month_cos`, `day`, `dayofweek`, `dayofweek_sin`, `dayofweek_cos`, `is_weekend`, `quarter`, `weekofyear`, `ordinal`
   - When sub-day timestamps are present: `hour`, `hour_sin`, `hour_cos`
4. **Write a feature audit** to `outputs/logs/<run_id>_profile.json` under `feature_audit`.
5. **Mention time feature detection in the report**: which columns were recognised, which features were generated, whether the target-signal audit shows a meaningful temporal pattern.

## What is FORBIDDEN

- Hard-coding any dataset-specific column names in the feature engineering logic.
- Skipping time feature extraction because a datetime column happens to be the row_id column.
- Adding the raw datetime string column as a model feature.

> Note for the current dataset: `period_id` is an **opaque, non-datetime** join key (verified in
> `spec_parse.json`). It must stay out of the model feature set and must **not** be parsed as a date —
> the scan above will correctly find it unparseable.
