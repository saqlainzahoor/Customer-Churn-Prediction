import joblib
from src.model_training import prepare_features

X_train, X_test, y_train, y_test, logistic_model, random_forest_model = prepare_features()

logistic_model.fit(X_train, y_train)

joblib.dump(logistic_model, "models/churn_logistic_model.pkl")

print("Model saved successfully.")
