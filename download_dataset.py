
import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

# Create the data folder
os.makedirs("data", exist_ok=True)

print("Downloading Credit Default dataset...")

try:
    # Fetch the dataset from UCI
    dataset = fetch_ucirepo(id=350)

    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    # Combine features and target
    df = pd.concat([X, y], axis=1)

    # Rename target column
    target_column = y.columns[0]
    df.rename(
        columns={target_column: "default_payment_next_month"},
        inplace=True
    )

    # Save dataset
    output_path = os.path.join("data", "credit_default.csv")
    df.to_csv(output_path, index=False)

    print("\nDataset downloaded successfully!")
    print("Dataset shape:", df.shape)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nTarget distribution:")
    print(df["default_payment_next_month"].value_counts())

    print("\nSaved at:", output_path)

except Exception as error:
    print("\nDataset download failed.")
    print("Error:", error)
