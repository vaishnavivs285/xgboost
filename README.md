# ❤️ Heart Disease Prediction using XGBoost  
### (Gain-based Splitting & Regularization)

---

## 1. Dataset Used

### 📊 Dataset Name  
**Heart Disease Dataset (UCI-inspired, Kaggle version)**

### 🔗 Source  
Kaggle – Heart Disease Dataset (originally derived from the UCI Heart Disease dataset)

### 🧾 Description  
This dataset contains clinical and diagnostic information used to predict whether a patient has heart disease.

- Rows: ~1000+ patient records  
- Target variable: `target`  
  - `0` → No heart disease  
  - `1` → Presence of heart disease  

### 🧠 Features (examples)

| Feature | Description |
|------|------------|
| age | Age of patient |
| sex | Gender (1 = male, 0 = female) |
| cp | Chest pain type |
| chol | Serum cholesterol |
| thalach | Maximum heart rate achieved |
| oldpeak | ST depression induced by exercise |
| ca | Number of major vessels |
| thal | Thalassemia type |

### 📌 Why this dataset?
- Real-world medical classification problem  
- Binary outcome (ideal for classification)  
- Moderate size (not toy, not extremely large)  
- Commonly used and interview-friendly  
- Requires generalization, not memorization  

---

## 2. Problem Statement

The objective of this project is to predict the presence of heart disease using patient clinical data while explicitly demonstrating how **XGBoost uses gain-based splitting and regularization to control overfitting and improve generalization**.

---

## 3. Why XGBoost?

XGBoost was chosen because:
- It is a boosting-based ensemble model  
- Trees are built sequentially to correct previous errors  
- It optimizes a differentiable loss function using gradients  
- Regularization is built directly into the objective function  
- Splits are decided using **gain**, not impurity  

This makes XGBoost suitable not only for performance, but also for understanding **model behavior and control**.

---

## 4. Why a Baseline Model?

A baseline XGBoost model was trained to:
- Observe model behavior without constraints  
- Understand overfitting risk  
- Create a reference point before regularization  

Baseline characteristics:
- Deeper trees  
- No penalty on splits (`gamma = 0`)  
- No penalty on leaf weights (`L1 = 0`, `L2 = 0`)  

This step helps justify why regularization is required.

---

## 5. Why Regularization Was Added

XGBoost is powerful, and powerful models tend to overfit.

Regularization was applied to:
- Restrict unnecessary splits  
- Penalize complex trees  
- Improve generalization on unseen data  

### Regularization Parameters

| Parameter | Purpose |
|--------|--------|
| gamma | Minimum gain required to make a split |
| reg_lambda (L2) | Penalizes large leaf weights |
| reg_alpha (L1) | Encourages sparsity |
| subsample | Row sampling to reduce variance |
| colsample_bytree | Feature sampling to reduce variance |

A split is allowed only when the **gain exceeds the regularization cost**.

---

## 6. Relationship Between Gain, Trees, and Regularization

1. XGBoost evaluates a possible split  
2. Gain is computed as reduction in loss  
3. Regularization penalty is applied  
4. If final gain ≤ 0, the split is rejected  

This ensures:
- Fewer but meaningful splits  
- Shallow trees  
- Stable learning behavior  

---

## 7. Model Evaluation Strategy

Accuracy alone is insufficient for classification problems.

Metrics used:
- Accuracy  
- Precision, Recall, F1-score  
- ROC–AUC  
- Confusion Matrix  

ROC–AUC was used to evaluate class separability across all thresholds, which is especially important in medical prediction tasks.

---

## 8. Gain-Based Feature Importance

Feature importance was computed using **gain**, which measures how much each feature contributes to reducing the loss function across all trees.

This aligns with XGBoost’s optimization objective and provides better interpretability than split counts.

---

## 9. Prediction Confidence Analysis

The predicted probability distribution was analyzed to:
- Detect overconfident predictions  
- Verify the effect of regularization  

The smooth distribution confirms controlled learning and reduced overfitting.

---

## 10. Workflow Summary

Dataset  
→ EDA & class balance check  
→ Baseline XGBoost  
→ Add regularization  
→ Train regularized model  
→ Evaluate with multiple metrics  
→ Interpret using gain & probability analysis  

Each step serves a clear purpose.

---

## 11. Key Learnings

- Gain-based splitting prevents unnecessary tree growth  
- Regularization is essential for powerful ensemble models  
- Accuracy alone is misleading  
- Interpretability and stability matter in real-world ML  

---

## 12. Future Improvements

- Cross-validation-based hyperparameter tuning  
- Cost-sensitive learning  
- Probability calibration  
- Comparison with LightGBM or CatBoost  

---

## 13. Conclusion

This project demonstrates how XGBoost combines gradient-based learning, gain-driven decisions, and regularization to build a robust and interpretable heart disease classification model.
