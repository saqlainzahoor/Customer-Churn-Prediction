import matplotlib.pyplot as plt

def plot_churn_by_contract(df):
    churn = (
        df.groupby("Contract")["Churn"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    churn.plot(kind="bar", title="Churn Rate by Contract Type")
    plt.ylabel("Churn Rate (%)")
    plt.xlabel("Contract Type")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
