import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load predictions
df = pd.read_csv("results/predictions.csv")

y_true = df["True Price"]

models = {
    "Image Only": df["Image Prediction"],
    "Metadata Only": df["Metadata Prediction"],
    "Text Only": df["Text Prediction"],
    "Simple Fusion": df["Simple Fusion Prediction"],
    "Attention Fusion": df["Attention Fusion Prediction"]
}

print("\n===== MODEL EVALUATION RESULTS =====")

for name, y_pred in models.items():
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{name}")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R2   : {r2:.4f}")