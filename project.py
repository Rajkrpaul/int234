
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Graph settings
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (9, 5)
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.titlepad"] = 14

blue = "#4C78A8"
teal = "#59A5A5"
orange = "#F28E2B"
red = "#E15759"
purple = "#9C6ADE"


# Load dataset
df = pd.read_csv(
    r"C:\Users\raj02\OneDrive\Desktop\int234project\train.csv"
)

print("\nDATASET INFORMATION")
print("-------------------")
print("Dataset shape:", df.shape)
print(df.head())
print(df.info())

print("\nMissing values:")
print(df.isnull().sum().sort_values(ascending=False).head(15))

print("\nDuplicate rows:", df.duplicated().sum())

print("\nSale price summary:")
print(df["SalePrice"].describe())


# Sale price distribution
plt.figure(figsize=(9, 5))

sns.histplot(
    df["SalePrice"],
    bins=40,
    kde=True,
    color=blue,
    edgecolor="white",
    alpha=0.8
)

plt.axvline(
    df["SalePrice"].mean(),
    color=red,
    linestyle="--",
    linewidth=2,
    label="Mean Price"
)

plt.axvline(
    df["SalePrice"].median(),
    color=orange,
    linestyle="--",
    linewidth=2,
    label="Median Price"
)

plt.title("Distribution of House Sale Prices")
plt.xlabel("Sale Price ($)")
plt.ylabel("Number of Houses")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("visualisations/sale_price_distribution.png", dpi=300)
plt.show()


# Data cleaning
df_clean = df.copy()

absent_cols = [
    "Alley", "BsmtQual", "BsmtCond", "BsmtExposure",
    "BsmtFinType1", "BsmtFinType2", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual",
    "GarageCond", "PoolQC", "Fence", "MiscFeature",
    "MasVnrType"
]

df_clean[absent_cols] = df_clean[absent_cols].fillna("None")

for col in df_clean.select_dtypes(include="object").columns:
    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

for col in df_clean.select_dtypes(include=np.number).columns:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

df_clean = df_clean.drop_duplicates()

print("\nDATA CLEANING")
print("-------------")
print("Missing values after cleaning:", df_clean.isnull().sum().sum())
print("Shape after cleaning:", df_clean.shape)

df_clean.to_csv("cleaned_house_prices.csv", index=False)


# EDA 1: House quality vs sale price
plt.figure(figsize=(9, 5))

sns.boxplot(
    data=df_clean,
    x="OverallQual",
    y="SalePrice",
    hue="OverallQual",
    palette="Set3",
    legend=False
)

plt.title("Effect of Overall House Quality on Sale Price")
plt.xlabel("Overall Quality Rating")
plt.ylabel("Sale Price ($)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("visualisations/quality_vs_price.png", dpi=300)
plt.show()


# EDA 2: Living area vs sale price
plt.figure(figsize=(9, 5))

sns.scatterplot(
    data=df_clean,
    x="GrLivArea",
    y="SalePrice",
    color=teal,
    alpha=0.7,
    edgecolor="white"
)

plt.title("Relationship Between Living Area and Sale Price")
plt.xlabel("Living Area (sq ft)")
plt.ylabel("Sale Price ($)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("visualisations/area_vs_price.png", dpi=300)
plt.show()


# EDA 3: Neighborhood prices
neighborhood_price = df_clean.groupby("Neighborhood")["SalePrice"].mean()
neighborhood_price = neighborhood_price.sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))

bars = plt.barh(
    neighborhood_price.sort_values().index,
    neighborhood_price.sort_values().values,
    color=sns.color_palette("viridis", 10),
    edgecolor="white"
)

for bar in bars:
    plt.text(
        bar.get_width() + 3000,
        bar.get_y() + bar.get_height() / 2,
        f"${bar.get_width():,.0f}",
        va="center",
        fontsize=9
    )

plt.title("Top 10 Neighborhoods by Average Sale Price")
plt.xlabel("Average Sale Price ($)")
plt.ylabel("Neighborhood")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("visualisations/neighborhood_prices.png", dpi=300)
plt.show()


# EDA 4: Correlation with sale price
corr = df_clean.select_dtypes(include=np.number).corr()

price_corr = corr["SalePrice"].drop("SalePrice")
price_corr = price_corr.sort_values(key=abs, ascending=False).head(10)

print("\nFEATURES CORRELATED WITH SALE PRICE")
print("-----------------------------------")
print(price_corr)

plt.figure(figsize=(9, 6))

bars = plt.barh(
    price_corr.sort_values().index,
    price_corr.sort_values().values,
    color=teal,
    edgecolor="white"
)

for bar in bars:
    plt.text(
        bar.get_width() + 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"{bar.get_width():.2f}",
        va="center",
        fontsize=9
    )

plt.title("Top 10 Features Related to Sale Price")
plt.xlabel("Correlation")
plt.ylabel("Feature")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("visualisations/price_correlation.png", dpi=300)
plt.show()


# Correlation heatmap
top_features = price_corr.index.tolist() + ["SalePrice"]
heatmap_data = df_clean[top_features].corr()

plt.figure(figsize=(10, 8))

sns.heatmap(
    heatmap_data,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
    square=True
)

plt.title("Correlation Heatmap of Important Features")
plt.tight_layout()
plt.savefig("visualisations/correlation_heatmap.png", dpi=300)
plt.show()


# Feature engineering
df_model = df_clean.copy()

df_model["HouseAge"] = df_model["YrSold"] - df_model["YearBuilt"]
df_model["YearsSinceRemodel"] = df_model["YrSold"] - df_model["YearRemodAdd"]
df_model["TotalSF"] = df_model["TotalBsmtSF"] + df_model["GrLivArea"]

df_model["TotalBathrooms"] = (
    df_model["FullBath"] + 0.5 * df_model["HalfBath"]
    + df_model["BsmtFullBath"] + 0.5 * df_model["BsmtHalfBath"]
)

df_model["TotalPorchSF"] = (
    df_model["WoodDeckSF"] + df_model["OpenPorchSF"]
    + df_model["EnclosedPorch"] + df_model["3SsnPorch"]
    + df_model["ScreenPorch"]
)

print("\nNEW FEATURES")
print("------------")
print(df_model[[
    "HouseAge", "YearsSinceRemodel", "TotalSF",
    "TotalBathrooms", "TotalPorchSF", "SalePrice"
]].head())

df_model.to_csv("house_prices_features.csv", index=False)


# Total area vs sale price
plt.figure(figsize=(9, 5))

sns.scatterplot(
    data=df_model,
    x="TotalSF",
    y="SalePrice",
    color=orange,
    alpha=0.7,
    edgecolor="white"
)

plt.title("Total House Area vs Sale Price")
plt.xlabel("Total Area (sq ft)")
plt.ylabel("Sale Price ($)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("visualisations/total_area_vs_price.png", dpi=300)
plt.show()


# House age vs sale price
plt.figure(figsize=(9, 5))

sns.scatterplot(
    data=df_model,
    x="HouseAge",
    y="SalePrice",
    color=purple,
    alpha=0.7,
    edgecolor="white"
)

plt.title("House Age vs Sale Price")
plt.xlabel("Age at Sale (years)")
plt.ylabel("Sale Price ($)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("visualisations/house_age_vs_price.png", dpi=300)
plt.show()


# Machine learning
X = df_model.drop(columns=["SalePrice", "Id"])
y = df_model["SalePrice"]

num_cols = X.select_dtypes(include=np.number).columns
cat_cols = X.select_dtypes(include="object").columns

num_process = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_process = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer([
    ("num", num_process, num_cols),
    ("cat", cat_process, cat_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nMACHINE LEARNING")
print("----------------")
print("Training rows:", X_train.shape[0])
print("Testing rows:", X_test.shape[0])


# Train four models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05,
        max_depth=3, random_state=42
    )
}

results = []
trained_models = {}

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocess", preprocess),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    results.append([name, mae, rmse, r2])
    trained_models[name] = pipeline


# Model comparison
results_df = pd.DataFrame(
    results, columns=["Model", "MAE", "RMSE", "R2"]
)

results_df = results_df.sort_values("RMSE").reset_index(drop=True)

print("\nMODEL COMPARISON")
print("----------------")
print(results_df.round(2).to_string(index=False))

results_df.to_csv("model_results.csv", index=False)


# Compare RMSE
plt.figure(figsize=(9, 5))

bars = plt.barh(
    results_df["Model"],
    results_df["RMSE"],
    color=["#59A5A5", "#4C78A8", "#F28E2B", "#9C6ADE"],
    edgecolor="white"
)

for bar in bars:
    plt.text(
        bar.get_width() + 500,
        bar.get_y() + bar.get_height() / 2,
        f"{bar.get_width():,.0f}",
        va="center",
        fontsize=9
    )

plt.title("Model Comparison Based on RMSE")
plt.xlabel("RMSE (lower is better)")
plt.ylabel("Machine Learning Model")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("visualisations/model_comparison.png", dpi=300)
plt.show()


# Compare R2
plt.figure(figsize=(9, 5))

bars = plt.barh(
    results_df["Model"],
    results_df["R2"],
    color=["#59A5A5", "#4C78A8", "#F28E2B", "#9C6ADE"],
    edgecolor="white"
)

for bar in bars:
    plt.text(
        bar.get_width() + 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"{bar.get_width():.3f}",
        va="center",
        fontsize=9
    )

plt.title("Model Comparison Based on R2 Score")
plt.xlabel("R2 Score (higher is better)")
plt.ylabel("Machine Learning Model")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("visualisations/model_r2_comparison.png", dpi=300)
plt.show()


# Save the best model
best_name = results_df.loc[0, "Model"]
best_model = trained_models[best_name]

joblib.dump(best_model, "best_house_price_model.pkl")

print("\nBEST MODEL")
print("----------")
print("Best model:", best_name)
print("MAE:", round(results_df.loc[0, "MAE"], 2))
print("RMSE:", round(results_df.loc[0, "RMSE"], 2))
print("R2:", round(results_df.loc[0, "R2"], 4))


# Actual vs predicted prices
best_pred = best_model.predict(X_test)

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    best_pred,
    alpha=0.7,
    color=teal,
    edgecolor="white",
    label="Predicted prices"
)

low = min(y_test.min(), best_pred.min())
high = max(y_test.max(), best_pred.max())

plt.plot(
    [low, high],
    [low, high],
    color=red,
    linestyle="--",
    linewidth=2,
    label="Perfect prediction"
)

plt.xlabel("Actual Sale Price ($)")
plt.ylabel("Predicted Sale Price ($)")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("visualisations/actual_vs_predicted.png", dpi=300)
plt.show()


# Feature importance
if best_name in ["Decision Tree", "Random Forest", "Gradient Boosting"]:

    feature_names = best_model.named_steps["preprocess"].get_feature_names_out()
    importance = best_model.named_steps["model"].feature_importances_

    feature_importance = pd.Series(
        importance,
        index=feature_names
    ).sort_values(ascending=False).head(15)

    plt.figure(figsize=(10, 6))

    feature_importance.sort_values().plot(
        kind="barh",
        color=purple,
        edgecolor="white"
    )

    plt.title("Top 15 Important Features in the Best Model")
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("visualisations/feature_importance.png", dpi=300)
    plt.show()


# Save predictions
predictions = pd.DataFrame({
    "ActualPrice": y_test,
    "PredictedPrice": best_pred
})

predictions.to_csv("best_model_predictions.csv", index=False)


# Project conclusion
print("\n" + "=" * 50)
print("                 PROJECT CONCLUSION")
print("=" * 50)

print("\n1. Dataset:")
print("The project analyzed", len(df_clean), "house records.")

print("\n2. Important factors:")
print(
    price_corr.index[0],
    "has the strongest correlation with sale price:",
    round(price_corr.iloc[0], 3)
)

print("\n3. Neighborhood analysis:")
print(
    neighborhood_price.index[0],
    "has the highest average sale price among the top 10 neighborhoods."
)

print("\n4. Model performance:")
print("The best performing model is", best_name)
print("Its average absolute prediction error is $", round(results_df.loc[0, "MAE"], 2))
print("Its RMSE is $", round(results_df.loc[0, "RMSE"], 2))
print("Its R2 score is", round(results_df.loc[0, "R2"], 3))

print("\n5. Final observation:")
print(
    "The results show that house quality, living area and other property "
    "features are useful for estimating house prices."
)

print(
    "The selected model explains approximately",
    round(results_df.loc[0, "R2"] * 100, 1),
    "% of the variation in sale prices in the test data."
)

print("\nAll results and visualisations have been saved.")
print("=" * 50)
