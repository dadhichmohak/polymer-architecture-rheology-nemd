import numpy as np

data=np.loadtxt("rheology_linear.dat")

gdot=data[:,0]
eta=data[:,1]

# eta = K * gdot^(n-1)

x=np.log10(gdot)
y=np.log10(eta)

m,b=np.polyfit(x,y,1)

n=m+1
K=10**b

print("\n===== POWER LAW FIT =====")
print(f"K = {K:.6f}")
print(f"n = {n:.6f}")
print(f"slope = {m:.6f}")
