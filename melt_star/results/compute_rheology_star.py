import numpy as np

gdots = [0.001,0.003,0.01,0.03,0.1]

print("# gdot    viscosity    N1")

for g in gdots:
    fn = f"../nemd/gdot_{g}/output/stress_vs_time.dat"

    data = np.loadtxt(fn, comments="#")

    pxy = data[:,1].mean()
    pxx = data[:,2].mean()
    pyy = data[:,3].mean()

    eta = -pxy/g
    N1  = pxx-pyy

    print(f"{g:<8g} {eta:<12.6f} {N1:<12.6f}")
