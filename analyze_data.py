
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/credit_default.csv")

print("DATASET OVERVIEW")
print("-" * 40)
print("Shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nDuplicate rows:", df.duplicated().sum())

target = "default_payment_next_month"

print("\nTarget distribution:")
print(df[target].value_counts())

print("\nTarget percentage:")
print((df[target].value_counts(normalize=True) * 100).round(2))

# Plot target distribution
sns.set_theme(style="whitegrid")
plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x=target
)

plt.title("Credit Default Distribution")
plt.xlabel("Default Next Month (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("data/target_distribution.png", dpi=150)
plt.show()

print("\nChart saved to data/target_distribution.png")
