
# SafeDrive Insurance: Motor Insurance Claim Prediction

A multi-stage data science project that predicts motor insurance claims and groups drivers into Low, Medium and High risk tiers.

## Project Structure

| File | What it covers |
|---|---|
| `01_data_preprocessing_eda_feature_engineering.ipynb` | Exploratory data analysis, data preprocessing and feature engineering |
| `02_ml_model_building_and_validation.ipynb` | Logistic regression baseline with Box-Tidwell tests, VIF filtering and stratified cross-validation |
| `03_automl_model_comparison_and_versioning.ipynb` | Featuretools feature generation, Optuna tuning, MLflow experiment tracking, Evidently drift reports and SHAP explainability |
| `dashboard_report.pdf` | 2-page risk dashboard: portfolio overview, risk tiers and key risk drivers |
| `prediction_output.csv` | Test-set predictions (11,719 policies) with predicted probability, class and risk tier |

## Key Results

| Metric (test set) | Logistic Regression | Tuned Decision Tree |
|---|---|---|
| ROC-AUC | 0.596 | 0.651 |
| Recall | 0.560 | 0.683 |

The tuned Decision Tree outperformed the logistic regression baseline on recall and ROC-AUC. Claims are rare (6.4% of policies), so the model gives a modest but real lift. High-risk drivers claim at about 10.2%, against 2.8% for Low-risk drivers.

## Tools Used

Python, pandas, scikit-learn, Featuretools, Optuna, MLflow, Evidently, SHAP

## Author

Arya Boricha | Mumbai, India
