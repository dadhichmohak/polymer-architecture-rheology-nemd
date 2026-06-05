import numpy as np
import pandas as pd

np.random.seed(42)

architectures = {
    "Linear": {
        "eta001":22.0715,
        "eta003":22.0054,
        "eta01":11.5954,
        "eta03":6.52423,
        "eta1":3.61959,
        "N1_001":0.0766664,
        "N1_003":0.22432,
        "N1_01":0.630471,
        "N1_03":1.47788,
        "N1_1":2.95120,
        "lambda_max":0.469606
    },

    "Star": {
        "eta001":38.3464,
        "eta003":23.2764,
        "eta01":13.3515,
        "eta03":7.71063,
        "eta1":4.20575,
        "N1_001":0.034846,
        "N1_003":0.136584,
        "N1_01":0.429792,
        "N1_03":1.036505,
        "N1_1":2.318499,
        "lambda_max":0.386658
    },

    "H-shaped": {
        "eta001":30.7012,
        "eta003":16.5747,
        "eta01":11.8830,
        "eta03":7.0967,
        "eta1":4.0617,
        "N1_001":0.06931,
        "N1_003":0.10850,
        "N1_01":0.49202,
        "N1_03":0.87630,
        "N1_1":2.22923,
        "lambda_max":0.393195
    }
}

rows = []

uid = 1

for arch in ["Linear","Star","H-shaped"]:

    for _ in range(12):

        base = architectures[arch]

        row = {"ID":f"U{uid:02d}",
               "TrueClass":arch}

        for k,v in base.items():

            noise = np.random.normal(
                loc=0.0,
                scale=0.07
            )

            row[k] = v*(1+noise)

        rows.append(row)

        uid += 1

# 4 harder borderline cases

pairs = [
    ("Linear","H-shaped"),
    ("Star","H-shaped"),
    ("Linear","Star"),
    ("Star","H-shaped")
]

for a,b in pairs:

    row = {"ID":f"U{uid:02d}",
           "TrueClass":f"{a}/{b}"}

    for k in architectures[a]:

        row[k] = (
            architectures[a][k]
            + architectures[b][k]
        )/2.0

    rows.append(row)
    uid += 1

df = pd.DataFrame(rows)

df.to_csv(
    "benchmark_unknowns.csv",
    index=False
)

print(df[["ID","TrueClass"]])
print("\nGenerated:",len(df),"samples")
