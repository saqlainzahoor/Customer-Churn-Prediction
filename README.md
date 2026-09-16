# Customer Churn Prediction

A machine learning classification project that predicts whether a customer is likely to churn based on customer demographics, services, contract details, and billing information.

## Project Overview

This project demonstrates an end-to-end machine learning workflow including data cleaning, exploratory data analysis, feature preprocessing, machine learning classification, model evaluation, model saving, and Streamlit deployment.

## Dataset

- 7,043 customer records
- 20 features after preprocessing
- Target: Churn
- 0 = No Churn
- 1 = Churn

## Machine Learning Models

### Logistic Regression

- Accuracy: 80.55%
- Precision: 65.72%
- Recall: 55.88%
- F1-Score: 60.40%
- ROC-AUC: 0.8421

### Random Forest

- Accuracy: 77.08%
- Precision: 55.92%
- Recall: 64.44%
- F1-Score: 59.88%
- ROC-AUC: 0.8200

Logistic Regression was selected as the primary model because it achieved better overall performance.

## Key Insights

- Month-to-month customers had the highest churn rate.
- Customers with shorter tenure were more likely to churn.
- Fiber optic customers showed a higher churn rate in this dataset.
- Electronic check customers showed a higher churn rate than other payment methods.

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app/app.py