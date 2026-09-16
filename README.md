# Customer Churn Prediction

An end-to-end machine learning project that predicts whether a customer is likely to churn based on customer demographics, services, contract details, and billing information.

## Live Demo

[Launch the Customer Churn Prediction App](https://customer-churn-prediction-ww4wjkbsmzdn2czh2zhz3w.streamlit.app/)

## GitHub Repository

[View the source code on GitHub](https://github.com/saqlainzahoor/Customer-Churn-Prediction)

## Project Overview

Customer churn is an important business problem because understanding which customers are at higher risk of leaving can support customer retention analysis.

This project demonstrates an end-to-end machine learning workflow:

* Data loading and cleaning
* Exploratory data analysis
* Feature preprocessing
* Categorical encoding
* Numerical feature scaling
* Model training
* Model evaluation
* Model serialization with Joblib
* Interactive Streamlit application
* Cloud deployment

## Dataset

The project uses the Telco Customer Churn dataset.

* 7,043 customer records
* 19 input features
* Target variable: `Churn`
* `0` = No Churn
* `1` = Churn

## Exploratory Data Analysis

Some notable patterns observed in the dataset:

* Month-to-month customers had a churn rate of 42.71%.
* One-year contract customers had a churn rate of 11.27%.
* Two-year contract customers had a churn rate of 2.83%.
* Fiber optic customers had a churn rate of 41.89%.
* DSL customers had a churn rate of 18.96%.
* Customers without internet service had a churn rate of 7.40%.
* Electronic check customers had a churn rate of 45.29%.

These are descriptive patterns in this dataset and should not be interpreted as proof that any individual feature independently causes churn.

## Machine Learning Models

Two classification models were evaluated:

### Logistic Regression

* Accuracy: 80.55%
* Precision: 65.72%
* Recall: 55.88%
* F1-Score: 60.40%
* ROC-AUC: 0.8421

### Random Forest

* Accuracy: 77.08%
* Precision: 55.92%
* Recall: 64.44%
* F1-Score: 59.88%
* ROC-AUC: 0.8200

Logistic Regression was used as the primary model for the deployed application based on the evaluation results above.

## Feature Preprocessing

The preprocessing pipeline includes:

* One-hot encoding for categorical variables
* Standard scaling for numerical variables
* `handle_unknown="ignore"` for unseen categorical values
* Logistic Regression classification

The preprocessing steps are included inside the saved scikit-learn pipeline so that training and prediction use the same transformations.

## Streamlit Application

The application allows users to enter customer information such as:

* Gender
* Senior Citizen status
* Partner and Dependents
* Tenure
* Phone service
* Internet service
* Online security and backup
* Device protection
* Tech support
* Streaming services
* Contract type
* Paperless billing
* Payment method
* Monthly charges
* Total charges

The application then returns:

* Churn prediction
* Churn probability
* Risk message

## Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Streamlit
* Joblib
* Git
* GitHub

## Project Structure

```text
Customer-Churn-Prediction/
│
├── app/
│   └── app.py
│
├── data/
│   ├── Telco-Customer-Churn.csv
│   └── cleaned_telco_churn.csv
│
├── models/
│   └── churn_logistic_model.pkl
│
├── notebooks/
│
├── src/
│   ├── __init__.py
│   ├── data_cleaning.py
│   ├── eda.py
│   ├── evaluation.py
│   ├── model_training.py
│   ├── save_model.py
│   └── train_model.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Run Locally

Clone the repository:

```bash
git clone https://github.com/saqlainzahoor/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app/app.py
```

## Model Saving

The trained Logistic Regression pipeline is saved with Joblib:

```text
models/churn_logistic_model.pkl
```

The Streamlit application loads this saved pipeline instead of retraining the model for every prediction.

## Future Improvements

Possible next improvements include:

* Hyperparameter tuning
* Cross-validation
* Threshold optimization for churn recall
* Additional model comparison
* Feature importance and model interpretability
* Automated testing
* Improved dashboard visualizations
* CI/CD integration

## Author

Saqlain Zahoor
