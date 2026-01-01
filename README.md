## Titanic Survival Prediction (From Scratch)

### Problem
Predict passenger survival using historical Titanic data.

### Approach
- Careful feature engineering (Title, FamilySize, HasCabin)
- Leakage-safe preprocessing
- Stratified train–test split
- Logistic Regression as an interpretable baseline

### Evaluation
- Metric: ROC-AUC
- Score: ~0.87 on test set

### Why Logistic Regression?
- Small dataset
- Interpretability
- Strong baseline before complex models

### Key Learnings
- Importance of stratification
- Group-wise median imputation
- Why accuracy is misleading for imbalanced data

### How to Run
```bash
python src/train_model.py
