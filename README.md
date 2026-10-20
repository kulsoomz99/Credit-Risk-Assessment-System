
# Credit Risk Assessment Using Machine Learning


## Author

**Kulsoom Zaidi**

## Project Overview

This project aims to build a robust machine learning model to predict the likelihood of loan default (`loan_status`). By analyzing historical customer data, including income, employment length, loan amount, interest rate, and credit history, the model helps financial institutions make safer and more informed lending decisions.

The project focuses on handling class imbalance, performing feature engineering, and comparing traditional statistical models (Logistic Regression) with ensemble methods (XGBoost). It also incorporates model interpretability using SHAP values and probability calibration.

## Key Features

- **Data Preprocessing:** Handling missing values, outlier detection, and duplicate removal.
- **Feature Engineering:** Preparing numerical and categorical features for machine learning.
- **Class Imbalance Handling:** Using `scale_pos_weight` in XGBoost and `class_weight` in Logistic Regression.
- **Pipeline Implementation:** Using `sklearn.pipeline` and `ColumnTransformer` to prevent data leakage.
- **Model Comparison:** Benchmarking Logistic Regression against XGBoost.
- **Hyperparameter Tuning:** Using `RandomizedSearchCV` to optimize XGBoost parameters.
- **Interpretability:** Using SHAP for global and local feature importance analysis.
- **Probability Calibration:** Applying sigmoid calibration to improve probability reliability.
- **Threshold Optimization:** Analyzing Precision-Recall curves to select an appropriate classification threshold.
- **Error Analysis:** Investigating false positives and false negatives.


## Methodology

### 1. Modeling Strategy

#### Logistic Regression (Baseline Model)

- Used as the baseline classification model.
- Applied standard scaling to numerical features.
- Encoded categorical variables using One-Hot Encoding.
- Used `class_weight` to address class imbalance.
- Evaluated using multiple classification metrics.

#### XGBoost (Primary Model)

- Used a tree-based gradient boosting classifier.
- Handled class imbalance using `scale_pos_weight`.
- Captured nonlinear relationships between features.
- Optimized model performance through hyperparameter tuning.

### 2. Hyperparameter Tuning

`RandomizedSearchCV` with 5-fold Stratified Cross-Validation was used to tune the XGBoost model.

The following hyperparameters were considered:

| Hyperparameter | Description |
|---|---|
| `n_estimators` | Number of boosting trees |
| `max_depth` | Maximum depth of each tree |
| `learning_rate` | Boosting learning rate |
| `subsample` | Fraction of samples used for each tree |
| `colsample_bytree` | Fraction of features used for each tree |
| `min_child_weight` | Minimum child-node weight |
| `gamma` | Minimum loss reduction required for a split |

### 3. Evaluation Metrics

Because loan default is a class-imbalanced classification problem, accuracy alone is insufficient.

The following metrics were used:

- **Accuracy:** Proportion of correct predictions.
- **Precision:** Proportion of predicted defaults that were actual defaults.
- **Recall:** Proportion of actual defaults correctly identified.
- **F1-Score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Measures the model's ability to distinguish between default and non-default cases.
- **PR-AUC:** Evaluates precision-recall performance, especially useful for imbalanced datasets.

## Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.82 | 0.56 | 0.79 | 0.65 | 0.87 |
| **XGBoost (Tuned)** | **0.93** | **0.88** | **0.80** | **0.83** | **0.94** |

### Key Findings

- **Higher Accuracy:** XGBoost achieved an accuracy of 93%.
- **Improved Precision:** XGBoost achieved a precision of 0.88 compared to 0.56 for Logistic Regression.
- **Comparable Recall:** Both models achieved similar recall, with XGBoost reaching 0.80.
- **Better F1-Score:** XGBoost achieved an F1-score of 0.83.
- **Strong Discrimination:** XGBoost achieved a ROC-AUC of 0.94.

Overall, XGBoost significantly outperformed the Logistic Regression baseline across the reported evaluation metrics.


## Probability Calibration and Threshold Optimization

### Probability Calibration

The final model was calibrated using sigmoid calibration to improve the reliability of its predicted probabilities.

Calibration helps ensure that predicted default probabilities better correspond to observed default frequencies.

### Threshold Optimization

Precision-Recall curves were analyzed to identify a suitable classification threshold.

- **Lower threshold:** Can increase default detection but may produce more false positives.
- **Higher threshold:** Can reduce false positives but may increase false negatives.

The optimal threshold depends on the lender's risk tolerance and the relative financial costs of prediction errors.

## Model Interpretability Using SHAP

SHAP (SHapley Additive exPlanations) was used to explain the model's predictions and identify important features.

### Global Feature Importance

The analysis identified the following features among the most influential predictors:

- `loan_int_rate`
- `person_emp_length`
- `loan_grade`

SHAP summary plots help illustrate how different feature values influence predicted default risk.

### Local Feature Explanations

SHAP waterfall plots explain individual predictions by showing how each feature contributes to moving the prediction away from the baseline toward a higher or lower estimated default risk.

These explanations help analysts understand model behavior and investigate individual cases.

## Error Analysis

The model's prediction errors were analyzed to understand their potential business impact.

| Error Type | Count | Interpretation |
|---|---:|---|
| False Positives (FP) | 153 | Safe customers incorrectly predicted to default |
| False Negatives (FN) | 278 | Customers who defaulted incorrectly predicted as safe |

### False Positives

False positives may result in creditworthy customers being incorrectly classified as risky. This can lead to rejected applications, lost business opportunities, and a poorer customer experience.

### False Negatives

False negatives occur when customers who eventually default are predicted to be safe. These errors can expose lenders to unexpected financial losses.

### Business Trade-Off

The classification threshold should balance the cost of missed defaults against the cost of incorrectly rejecting creditworthy customers.





