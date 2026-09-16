import pandas as pd

DATA_PATH = "data/Telco-Customer-Churn.csv"

def load_and_clean_data():
    df = pd.read_csv(DATA_PATH)

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    df = df.drop(columns=["customerID"])

    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    return df
