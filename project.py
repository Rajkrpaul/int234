
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

missing = df.isnull().sum()

missing = missing[missing > 0].sort_values(ascending=False)

print(missing)


df_clean = df.copy()

# Columns where missing means the feature is absent
absent_cols = [
    "Alley", "BsmtQual", "BsmtCond", "BsmtExposure",
    "BsmtFinType1", "BsmtFinType2", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual",
    "GarageCond", "PoolQC", "Fence", "MiscFeature"
]

for col in absent_cols:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].fillna("None")

# Fill remaining categorical columns
for col in df_clean.select_dtypes(include="object").columns:
    if df_clean[col].isnull().sum() > 0:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

# Fill remaining numerical columns
for col in df_clean.select_dtypes(include=np.number).columns:
    if df_clean[col].isnull().sum() > 0:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

print("Missing values after cleaning:", df_clean.isnull().sum().sum())


print("Duplicate rows:", df_clean.duplicated().sum())

df_clean = df_clean.drop_duplicates()

print("Shape after removing duplicates:", df_clean.shape)


print(df_clean.dtypes)

print("\nUnique values in categorical columns:")
for col in df_clean.select_dtypes(include="object").columns:
    print(col, ":", df_clean[col].nunique())


df_clean.to_csv("cleaned_house_prices.csv", index=False)

print("Cleaned dataset saved.")

