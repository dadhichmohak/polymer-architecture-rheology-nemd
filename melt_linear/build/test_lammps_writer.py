with open("test.data","w") as f:
    f.write("LAMMPS data file\n\n")
    f.write("2 atoms\n")
    f.write("1 bonds\n\n")

    f.write("1 atom types\n")
    f.write("1 bond types\n\n")

    f.write("0 10 xlo xhi\n")
    f.write("0 10 ylo yhi\n")
    f.write("0 10 zlo zhi\n\n")

    f.write("Masses\n\n")
    f.write("1 1.0\n\n")

    f.write("Atoms\n\n")
    f.write("1 1 1 1.0 1.0 1.0\n")
    f.write("2 1 1 2.0 1.0 1.0\n\n")

    f.write("Bonds\n\n")
    f.write("1 1 1 2\n")
