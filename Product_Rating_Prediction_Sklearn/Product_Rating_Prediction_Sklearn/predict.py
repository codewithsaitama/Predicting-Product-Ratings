from pathlib import Path
import joblib
import pandas as pd

BASE = Path(__file__).resolve().parent
model = joblib.load(BASE / "product_rating_model.pkl")

sample = pd.DataFrame([{
    "product_quality": 9,
    "value_for_money": 8,
    "delivery_rating": 9,
    "customer_support": 8,
    "verified_purchase": 1
}])

prediction = float(model.predict(sample)[0])
prediction = max(1.0, min(5.0, prediction))
print(f"Predicted Product Rating: {prediction:.2f} / 5")
