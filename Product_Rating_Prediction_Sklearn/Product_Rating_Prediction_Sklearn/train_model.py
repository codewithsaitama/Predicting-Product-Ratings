from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path(__file__).resolve().parent
df = pd.read_csv(BASE / "data" / "product_ratings.csv")

X = df.drop(columns=["product_rating"])
y = df["product_rating"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print(f"MAE: {mean_absolute_error(y_test, pred):.2f}")
print(f"RMSE: {mean_squared_error(y_test, pred) ** 0.5:.2f}")
print(f"R2 Score: {r2_score(y_test, pred):.4f}")

joblib.dump(model, BASE / "product_rating_model.pkl")

importance = pd.Series(model.feature_importances_, index=X.columns).sort_values()
importance.plot(kind="barh", title="Product Rating Feature Importance")
plt.tight_layout()
plt.savefig(BASE / "feature_importance.png", dpi=150)
print("Model and chart saved successfully.")
