from src.model_training import prepare_features

X_train, X_test, y_train, y_test, model = prepare_features()

model.fit(X_train, y_train)

print("Model training completed successfully.")
