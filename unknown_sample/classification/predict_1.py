import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

db_file = ROOT / "database" / "features.csv"
benchmark_file = ROOT / "unknown_sample" / "benchmark" / "benchmark_unknowns.csv"

db = pd.read_csv(db_file)
bench = pd.read_csv(benchmark_file)

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

results = []

correct = 0
total = 0

print("\n====================================")
print(" UNKNOWN POLYMER CLASSIFICATION")
print("====================================\n")

for _, sample in bench.iterrows():

    best_arch = None
    best_dist = 1e99

    for _, row in db.iterrows():

        d = 0.0

        for f in feature_cols:

            scale = db[f].max() - db[f].min()

            if scale < 1e-12:
                scale = 1.0

            d += ((sample[f] - row[f]) / scale) ** 2

        d = np.sqrt(d)

        if d < best_dist:
            best_dist = d
            best_arch = row["Architecture"]

    truth = sample["TrueClass"]

    if "/" not in truth:
        total += 1

        if best_arch == truth:
            correct += 1

    results.append([
        sample["ID"],
        truth,
        best_arch,
        best_dist
    ])

    print(
        f"{sample['ID']:4s} | "
        f"True = {truth:10s} | "
        f"Pred = {best_arch:10s} | "
        f"Dist = {best_dist:.4f}"
    )

accuracy = 100.0 * correct / total

print("\n====================================")
print(f"Accuracy = {accuracy:.2f}%")
print("====================================")

df = pd.DataFrame(
    results,
    columns=[
        "ID",
        "TrueClass",
        "Predicted",
        "Distance"
    ]
)

df.to_csv(
    "classification_results.csv",
    index=False
)

with open("classification_summary.txt", "w") as f:

    f.write("Unknown Polymer Classification\n")
    f.write("=============================\n\n")

    f.write(
        f"Accuracy = {accuracy:.2f}%\n\n"
    )

    f.write(
        df.to_string(index=False)
    )

print("\nSaved:")
print("classification_results.csv")
print("classification_summary.txt")
