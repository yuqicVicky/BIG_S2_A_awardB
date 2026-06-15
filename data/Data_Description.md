# Iris Dataset Description

## Overview

The classic **Iris** dataset (Fisher, 1936). Each row is one iris flower described by
four morphological measurements, labelled with one of three species. The task is a
**multi-class classification** problem: predict the species from the four measurements.

## Files

| File | Rows | Purpose |
|------|------|---------|
| `train.csv` | 120 | Training split (stratified by class) |
| `val.csv`   | 30  | Validation split (stratified by class) |

The 80/20 split is stratified on `label`, so the three classes are balanced
(40 per class in train, 10 per class in val).

## Columns

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `sepal length (cm)` | float | cm | Length of the sepal |
| `sepal width (cm)`  | float | cm | Width of the sepal |
| `petal length (cm)` | float | cm | Length of the petal |
| `petal width (cm)`  | float | cm | Width of the petal |
| `label`   | int  | — | Encoded target class: `0`, `1`, or `2` |
| `species` | str  | — | Human-readable target class name |

## Target

- **Target column:** `label` (or equivalently `species`)
- **Task type:** multi-class classification (3 classes)
- **Class mapping:**

  | label | species |
  |-------|------------|
  | 0 | setosa |
  | 1 | versicolor |
  | 2 | virginica |

`label` and `species` encode the same information; use one as the target and drop the
other to avoid leakage.

## Notes

- 4 numeric features, no missing values.
- Features are positive, continuous measurements in centimetres.
- Classes are perfectly balanced after the stratified split.
- `setosa` is linearly separable from the other two; `versicolor` and `virginica`
  overlap and are the main source of classification error.

## Recommended Metric

Accuracy (balanced classes) or macro-averaged F1.
