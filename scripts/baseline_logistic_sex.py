"""Baseline: logistic regression on Sex only. Titanic survival prediction.

Output: submission_base.csv (PassengerId, Survived) in test row order.
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

ROOT = "/Users/yuqic/Documents/Claude/Projects/STAI-X challenge/award-b-repo"

train = pd.read_csv(f"{ROOT}/data/train.csv")
test = pd.read_csv(f"{ROOT}/data/test.csv")

# Single feature: Sex (encode female=1, male=0)
X = (train["Sex"] == "female").astype(int).to_frame("is_female")
y = train["Survived"]
X_test = (test["Sex"] == "female").astype(int).to_frame("is_female")

clf = LogisticRegression()
clf.fit(X, y)

cv = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
print(f"5-fold CV accuracy: {cv.mean():.4f} +/- {cv.std():.4f}")

preds = clf.predict(X_test)
out = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": preds.astype(int)})
out.to_csv(f"{ROOT}/submission_base.csv", index=False)
print(f"Wrote submission_base.csv: {len(out)} rows, predicted survivors: {out['Survived'].sum()}")
