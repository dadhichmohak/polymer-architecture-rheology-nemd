import numpy as np

# Read:
# gdot viscosity N1

data = np.loadtxt("rheology_h.dat", skiprows=1)

gdot = data[:,0]
eta  = data[:,1]

coef = np.polyfit(
    np.log(gdot),
    np.log(eta),
    1
)

n = coef[0] + 1
K = np.exp(coef[1])

print("Architecture : H-shaped Melt")
print(f"K            : {K:.6f}")
print(f"n            : {n:.6f}")

with open("powerlaw_fit.txt","w") as f:
    f.write("Architecture : H-shaped Melt\n")
    f.write(f"K            : {K:.6f}\n")
    f.write(f"n            : {n:.6f}\n")