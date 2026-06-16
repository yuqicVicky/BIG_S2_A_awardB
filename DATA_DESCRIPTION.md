# Titanic — Binary Survival Prediction

## Task
Binary classification: predict whether each passenger survived the sinking of the Titanic.

## Files
- `data/train.csv` — 891 rows with ground-truth `Survived` labels (training set)
- `data/test.csv` — 418 rows without labels (prediction target)
- `data/gender_submission.csv` — sample submission showing the required format (`PassengerId`, `Survived`)

## Columns (train.csv)
| Column | Description |
|--------|-------------|
| PassengerId | Unique passenger identifier (row ID) |
| Survived | **Target** — 0 = did not survive, 1 = survived |
| Pclass | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) — proxy for socioeconomic status |
| Name | Passenger name (includes title) |
| Sex | Passenger sex (male / female) |
| Age | Age in years (fractional if < 1; estimated if xx.5) |
| SibSp | Number of siblings / spouses aboard |
| Parch | Number of parents / children aboard |
| Ticket | Ticket number |
| Fare | Passenger fare |
| Cabin | Cabin number (sparse) |
| Embarked | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |

## Target variable
- **Column:** `Survived`
- **Type:** binary (0 / 1)
- **Row ID column:** `PassengerId`

## Evaluation metric
Binary classification accuracy (fraction of correct predictions). Standard Kaggle Titanic metric.

## Notes
- Age and Cabin have missing values.
- Embarked has 2 missing values in train.
- No time dimension; i.i.d. tabular data.
- Sample submission (`gender_submission.csv`) shows format: two columns `PassengerId,Survived` in test order.
