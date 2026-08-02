import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ===========================
# Read Dataset
# ===========================

df = pd.read_csv("dataset/car_data.csv")

print(df.head())
print(df.info())
print(df.isnull().sum())

# ===========================
# Data Cleaning
# ===========================

df = df.drop_duplicates()
df = df.dropna()

# ===========================
# Features and Target
# ===========================

features = [
    "Brand",
    "Model",
    "Year",
    "Engine_Size",
    "Fuel_Type",
    "Transmission",
    "Mileage",
    "Doors",
    "Owner_Count"
]

x = df[features]
y = df["Price"]

# ===========================
# Train Test Split
# ===========================

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

# ===========================
# Preprocessing
# ===========================

numerical = [
    "Year",
    "Engine_Size",
    "Mileage",
    "Doors",
    "Owner_Count"
]

categorical = [
    "Brand",
    "Model",
    "Fuel_Type",
    "Transmission"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
    ]
)

# ===========================
# Models
# ===========================

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(
        random_state=42,
        max_depth=10
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
}

results = {}
best_model = None
best_name = ""
best_score = -999

for name, model in models.items():

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipe.fit(x_train, y_train)

    y_pred = pipe.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results[name] = [rmse, r2]

    print("\n", name)
    print("RMSE:", rmse)
    print("R2 Score:", r2)

    if r2 > best_score:
        best_score = r2
        best_model = pipe
        best_name = name
        best_prediction = y_pred

# ===========================
# Model Comparison
# ===========================

results_df = pd.DataFrame(
    results,
    index=["RMSE", "R2 Score"]
).T

print("\nModel Comparison")
print(results_df)

# ===========================
# Save Model
# ===========================

joblib.dump(best_model, "model/car_price_model.pkl")

print("\nBest Model:", best_name)
print("Model saved successfully.")

# ===========================
# Actual vs Predicted
# ===========================

plt.figure(figsize=(7, 5))

plt.scatter(y_test, best_prediction)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted")

plt.tight_layout()
plt.show()

# ===========================
# Model Results
# ===========================

plt.figure(figsize=(8, 5))

sns.barplot(
    x=results_df.index,
    y=results_df["R2 Score"]
)

plt.xticks(rotation=15)
plt.title("Model R2 Score")
plt.tight_layout()
plt.show()
