import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_data(filepath="data/sales_data.csv"):
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        return None

    if df.empty:
        print("Error: The DataFrame is empty after loading data.")
        return None

    df = df.dropna()

    if df.empty:
        print("Error: All rows were dropped due to NaN values.")
        return None

    try:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["datetime"] = df["datetime"].astype(int) / 10**9
    except Exception as e:
        print(f"Error processing datetime column: {e}")
        return None

    label_encoders = {}
    for col in ["cash_type", "card", "coffee_name"]:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        else:
            print(f"Warning: Column '{col}' not found in the dataset.")

    if "money" not in df.columns:
        print("Error: Target column 'money' is missing.")
        return None

    try:
        X = df.drop(columns=["date", "money"])
        y = df["money"]
    except Exception as e:
        print(f"Error defining X and y: {e}")
        return None

    return train_test_split(X, y, test_size=0.2, random_state=42)
