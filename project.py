
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\raj02\OneDrive\Desktop\int234project\train.csv")

print(df.shape)
print(df.head())
print(df.info())

print("\nMissing values:")
print(df.isnull().sum().sort_values(ascending=False).head(15))

print("\nDuplicate rows:", df.duplicated().sum())

print("\nSale price summary:")
print(df["SalePrice"].describe())

plt.figure(figsize=(8, 5))
sns.histplot(df["SalePrice"], bins=40, kde=True)
plt.title("House Sale Price Distribution")
plt.xlabel("Sale Price")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(r"C:\Users\raj02\OneDrive\Desktop\int234project\visualisations/sale_price_distribution.png")
plt.show()
