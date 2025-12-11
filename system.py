# -------------------------- imports and settings ---------------------
import numpy as np
import matplotlib.pyplot as plt
import scipy
import random
import os 


# -------------------------- generate model ---------------------
class Model:
    """generate model for further Monte Carlo simulations"""
    def __init__(self, n_particle, density, work_dir, copies=1):
        self.n_particle = n_particle
        self.density = density
        self.work_dir = work_dir
        self.copies = copies

    def generate_model(self):
        side = (self.n_particle / self.density)**(1/3)
        model_files = []

        # Random initial configuration away from walls
        for i in range(1, self.copies+1):
            coords = np.random.uniform(0.1*side, 0.9*side, (self.n_particle, 3))
            filename = f"{self.n_particle}_{self.density}_{i}.txt"
            model = f"{self.work_dir}/{filename}"

            with open(model, "w") as box:
                box.write("BASIC INFORMATION\n")
                box.write(f"Total number of particles: {self.n_particle}\n")
                box.write(f"Density of model: {self.density}\n\n")

                box.write("BOX DIMENSION\n")
                box.write(f"0 {side:.6f} xlo xhi\n")
                box.write(f"0 {side:.6f} ylo yhi\n")
                box.write(f"0 {side:.6f} zlo zhi\n\n")

                box.write("PARTICLES\n")
                for i in range(self.n_particle):
                    x,y,z = coords[i]
                    box.write(f"{i+1} {x:.6f} {y:.6f} {z:.6f}\n")
            model_files.append(model)
            print(f"Created random model: {model}")
        print(model_files)
        return model_files

if __name__ == "__main__":
    coordi = Model(n_particle=8, density=0.064, 
                   work_dir="/path/where/you/want/to/save/model/file",
                   copies=10)
    coordi.generate_model()

