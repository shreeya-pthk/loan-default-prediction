# 🏦 Loan Default & Loss Prediction
### BA-64037-001 Advanced Data Mining & Predictive Analytics
**Kent State University — Group 5 | May 2025**

![Language](https://img.shields.io/badge/Language-R-276DC3?style=flat-square&logo=r&logoColor=white)
![Model](https://img.shields.io/badge/Classifier-Random%20Forest-2E7D32?style=flat-square)
![Model](https://img.shields.io/badge/Regressor-XGBoost-FF6F00?style=flat-square)
![MAE](https://img.shields.io/badge/Regressor%20MAE-5.2261-1565C0?style=flat-square)
![Recall](https://img.shields.io/badge/Default%20Recall-60.95%25-6A1B9A?style=flat-square)
![AUC](https://img.shields.io/badge/AUC-0.6212-00838F?style=flat-square)

---

## 📌 Overview

Banks and financial institutions lose billions annually to loan defaults.
This project builds a **two-stage machine learning pipeline** that goes beyond
simple binary classification ,it predicts both **whether** a loan will default
and **how large** the resulting financial loss will be.

The pipeline bridges traditional credit risk management with
asset management by quantifying both default probability and loss severity,
evaluated using **Mean Absolute Error (MAE)**.

---

## 👥 Group Members

| Name |
|---|
| **Shreeya Pathak** | 
| **Sri Anu Vunnam** | 
| **Inn Kyung Seo** | 
---

## 📊 Dataset

| Attribute | Detail |
|---|---|
| Source | Course portal (train_v3.csv / test_no_lossv3.csv) |
| Training records | 80,000 loans |
| Features | 763 anonymised variables (f1 – f776) |
| Defaulted loans | 7,379 (9.22%) |
| Non-defaulted loans | 72,621 (90.78%) |
| Target variable | `loss` — financial loss amount (0 if no default) |
| Test set | 25,471 loans (no loss labels) |

> ⚠️ **Class imbalance:** Only 9.22% of loans defaulted.
> A naive model predicting "no default" for everything achieves 90.78% accuracy
> but is completely useless. Careful handling was required.

---

## 🤖 Models

### Model 1 — Random Forest Classifier
Predicts whether a loan will default (binary: Yes / No)

| Parameter | Value | Reason |
|---|---|---|
| `ntree` | 300 | Ensemble stability |
| `mtry` | 13 (√p) | Standard rule-of-thumb |
| `classwt` | `c("0"=1, "1"=5)` | Penalise missed defaults 5× |
| `threshold` | 0.068 (Youden's method) | Maximises Sensitivity + Specificity |

**Why Random Forest?**
Handles class imbalance, captures non-linear relationships between
the 180 LASSO-selected features, and provides interpretable feature importance.

---

### Model 2 — XGBoost Regressor
Predicts the loss amount for loans flagged as defaults.
**Trained exclusively on 7,379 defaulted loans.**

| Parameter | Value | Reason |
|---|---|---|
| `objective` | reg:squarederror | Continuous loss prediction |
| `eval_metric` | mae | Matches grading metric directly |
| `learning_rate` | 0.03 | Slower but more precise convergence |
| `max_depth` | 4 | Controls tree complexity |
| `subsample` | 0.80 | Reduces overfitting |
| `colsample_bytree` | 0.80 | Feature subsampling per tree |

**Why XGBoost over linear regression?**
Financial loss data is non-linear. XGBoost captures feature interactions
(e.g., loan amount × credit utilisation) that linear models cannot model.

**Top predictive features:** f766 (strongest), f67, f229

---

## 📈 Results

### Classification Metrics (Random Forest)

| Metric | Value |
|---|---|
| Accuracy | 56.65% |
| Precision | 12.39% |
| **Recall (Sensitivity)** | **60.95%** |
| F1 Score | 0.2059 |
| **AUC** | **0.6212** |

> The low precision with high recall is intentional — the model is tuned
> to catch as many real defaults as possible, accepting more false alarms.
> In banking, **missing a default is far more costly than a false alarm.**

### Regression Metrics (XGBoost)

| Metric | Value |
|---|---|
| **MAE** | **5.2261** |
| RMSE | 10.4407 |

### Test Set Predictions

| Category | Count | Percentage |
|---|---|---|
| Total test loans | 25,471 | 100% |
| Predicted defaults | 11,424 | 44.9% |
| Predicted non-defaults | 14,047 | 55.1% |
| Rows with loss > 0 | 11,421 | — |
| Loss range | 0 – 54.19 | Mean: 3.69 |

> 📝 The predicted default rate (44.9%) is higher than the training
> rate (9.22%) due to the intentionally low threshold (0.068),
> which prioritises recall over precision.

### Confusion Matrix (Validation Set)

| | Predicted No Default | Predicted Default |
|---|---|---|
| **Actual No Default** | 8,165 ✅ | 6,359 ❌ |
| **Actual Default** | 576 ❌ | 899 ✅ |

---

## 🔍 Key Insights

1. **f766 is the strongest predictor** — consistently ranked #1 across
   correlation analysis, LASSO coefficients, and XGBoost feature importance

2. **Class imbalance is the core challenge** — with 9.22% defaults,
   `classwt` and low threshold tuning were critical to achieving
   meaningful recall (60.95%)

3. **XGBoost underestimates extreme losses (>50)** — right-skewed
   distribution means rare high-loss cases are underrepresented in
   training data; residuals grow negative at high actual values

4. **Two-model separation is essential** — a single regression on all
   80,000 loans is dominated by 72,621 zeros; separating classification
   from regression allows each model to specialise

5. **Recall vs precision trade-off** — threshold of 0.068 maximises
   default detection but inflates false positives, a known limitation
   documented in the report



