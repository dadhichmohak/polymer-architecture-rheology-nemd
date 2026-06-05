import numpy as np

gdot = np.array([0.001,0.003,0.01,0.03,0.1])

eta = np.array([
38.346392,
23.276373,
13.351538,
7.710630,
4.205754
])

coef = np.polyfit(np.log(gdot), np.log(eta), 1)

n = coef[0] + 1
K = np.exp(coef[1])

print("Architecture : Star Melt")
print(f"K            : {K:.6f}")
print(f"n            : {n:.6f}")
