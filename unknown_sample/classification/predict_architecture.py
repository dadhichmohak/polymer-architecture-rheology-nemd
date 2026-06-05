import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

db_file = ROOT / "database" / "features.csv"
unknown_file = ROOT / "unknown_sample" / "provided_data" / "unknown_features.csv"

db = pd.read_csv(db_file)
unknown = pd.read_csv(unknown_file)

u = unknown.iloc[0]

feature_cols = [
    "eta001",
    "eta003",
    "eta01",
    "eta03",
    "eta1",
    "N1_001",
    "N1_003",
    "N1_01",
    "N1_03",
    "N1_1",
    "lambda_max"
]

print("\n=== Unknown Sample Classification ===\n")

best_arch = None
best_dist = 1e99

for _, row in db.iterrows():

    d = 0.0

    for f in feature_cols:

        scale = db[f].max() - db[f].min()

        if scale < 1e-12:
            scale = 1.0

        d += ((u[f] - row[f]) / scale) ** 2

    d = np.sqrt(d)

    print(
        f"{row['Architecture']:10s} "
        f"distance = {d:.6f}"
    )

    if d < best_dist:
        best_dist = d
        best_arch = row["Architecture"]

# ----------------------------------------
# Unknown detection
# ----------------------------------------

UNKNOWN_THRESHOLD = 1.5

if best_dist > UNKNOWN_THRESHOLD:
    final_prediction = "UNKNOWN_ARCHITECTURE"
else:
    final_prediction = best_arch

print("\nPredicted Architecture:")
print(final_prediction)


print("\nSimilarity Score:")
print(f"{best_dist:.6f}")

with open("prediction.txt", "w") as f:
    f.write("Unknown Polymer Classification\n")
    f.write("==============================\n\n")
    f.write(f"Predicted Architecture : {final_prediction}\n")
    f.write(f"Distance Score         : {best_dist:.6f}\n")

print("\nSaved: prediction.txt")