# Titanic Survival Prediction

A beginner machine learning project predicting passenger survival on the Titanic using the [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic) dataset.

## Result
**Public Score: 0.76076** (Top ~80th percentile for first attempt)

## Tech Stack
- Python
- Pandas
- Scikit-learn
- Kaggle Notebooks

## Approach

### 1. Data Preprocessing
- Dropped irrelevant columns: `Name`, `Cabin`, `Ticket`
- Handled missing values:
  - `Age` → filled with median using `SimpleImputer`
  - `Embarked` → filled with most common value (`S`)
  - `Fare` → filled with median (test set only)

### 2. Feature Engineering
- Encoded categorical columns (`Sex`, `Embarked`) using `pd.get_dummies()`

### 3. Model
- Used `RandomForestClassifier` from scikit-learn
- `random_state=1` for reproducibility

## Files
| File | Description |
|------|-------------|
| `titanic.ipynb` | Main notebook with full code |
| `submission.csv` | Final predictions submitted to Kaggle |

## What I Learned
- Handling missing values with `fillna()` and `SimpleImputer`
- Encoding categorical features with `pd.get_dummies()`
- Difference between `fit_transform()` (train) and `transform()` (test) — avoiding **data leakage**
- Difference between Regression and Classification problems
- End-to-end ML workflow: data cleaning → preprocessing → model training → submission

## Future Improvements
- [ ] Feature engineering — extract `Title` from `Name` column
- [ ] Try XGBoost for better accuracy
- [ ] Use cross-validation for more reliable evaluation
