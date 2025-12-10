# -------------------------- imports and settings ---------------------
import os
import numpy as np
import random


# ------------------------- run Monte Carlo simulation ----------------
class MCSimulation:
    """conduct Monte Carlo simulation based on systems"""
    def __init__(self, model, epsilon, sigma, 
                 cutoff, n_step, temperature, kB, delta,
                 save_interval, log_filename=None):
        self.model = model
        self.epsilon = epsilon
        self.sigma = sigma
        self.cutoff = cutoff
        self.n_step = n_step
        self.temperature = temperature
        self.kB = kB
        self.delta = delta
        self.save_interval = save_interval
        model_path = os.path.dirname(model)
        if log_filename is None:
            log_base = os.path.splitext(os.path.basename(model))[0]
            self.log_filename = os.path.join(model_path, f"{log_base}_{self.n_step}_log.txt")
        else:
            self.log_filename = os.path.join(model_path, log_filename)


    # ----------- SIMULATION FEATURES ------------
    def load_model(self):
        """load and read model file if specified"""
        particles = []
        ids = []
        with open(self.model) as M:
            reading = False
            for line in M:
                # extract particle number
                stripped = line.strip()
                if line.startswith("Total number of particles"):
                    n_particles = int(line.split(":")[1].strip())

                # extract box dimension
                parts = stripped.split()
                if len(parts) >= 4 and parts[-2] == "xlo" and parts[-1] == "xhi":
                    b_min = float(parts[0])
                    b_max = float(parts[1])
            
                # extract particle coordinates
                if stripped == "PARTICLES":
                    reading = True
                    continue
                if reading:
                    if stripped == "" or not stripped[0].isdigit():
                        break
                    id, x, y, z = stripped.split()
                    ids.append(int(id))
                    particles.append([float(x), float(y), float(z)])
        coords = np.array(particles)
        coords = self.apply_pbc(coords, b_min, b_max)
        
        # Sort by ID to ensure consistent ordering (optional but safer)
        # If IDs are already sequential, this has no effect
        sorted_data = sorted(zip(ids, coords), key=lambda x: x[0])
        ids = [item[0] for item in sorted_data]
        coords = np.array([item[1] for item in sorted_data])
        
        return n_particles, b_min, b_max, coords, ids

    def minimum_image(self, vec, box_length):
        """apply minimum image convention to displacement vectors"""
        return vec - np.round(vec/box_length)*box_length
    
    def apply_pbc(self, coord, b_min, b_max):
        """apply periodic boundary condition on particles in the box"""
        box_length = b_max - b_min
        return b_min + ((coord-b_min)%box_length)
    
    def compute_pair_e(self, dist):
        """principles to compute correct pair energy"""
        if dist <= 1e-8:
            return 0
        if dist >= self.cutoff:
            return 0
        return 4*self.epsilon*((self.sigma/dist)**12 - (self.sigma/dist)**6)
    
    def compute_total_e(self, coords, b_min, b_max):
        """compute total energy of the system before moving particles"""
        E = 0
        n_particles = len(coords)
        box_length = b_max - b_min
        for i in range(n_particles):
            for j in range(i+1, n_particles):
                rij = coords[i] - coords[j]
                rij = self.minimum_image(rij, box_length)
                dist = np.linalg.norm(rij)
                E += self.compute_pair_e(dist)
        return E
    
    def compute_local_deltaE(self, coords, pid, old_pos, new_pos, b_min, b_max):
        """displace a particle and compute local potential energy change"""
        n_particles = len(coords)
        E_before = 0
        E_after = 0
        box_length = b_max - b_min
        for j in range(n_particles):
            if j == pid:
                continue
            # compute distance and potential energy
            rij_old = coords[j] - old_pos
            rij_old = self.minimum_image(rij_old, box_length)
            dist_old = np.linalg.norm(rij_old)
            E_before += self.compute_pair_e(dist_old)

            rij_new = coords[j] - new_pos
            rij_new = self.minimum_image(rij_new, box_length)
            dist_new = np.linalg.norm(rij_new)
            E_after += self.compute_pair_e(dist_new)
        return E_after - E_before
    

    # ----------- DATA LOGGING FEATURES ------------
    def save_log(self, step, sys_energy, coords, ids):
        """output log file during simulation for analysis"""
        with open(self.log_filename, "a") as log:
            log.write(f"STEP {step}\n")
            log.write(f"ENERGY {sys_energy:.6f}\n")
            for id, coord in zip(ids, coords):
                log.write(f"{id} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")
            log.write(f"\n")
    

    # ----------- PRIMARY SIMULATION LOOP -----------
    def run_simulation(self):
        """run Monte Carlo simulation for specified steps"""
        # initialize simulation conditions
        n_particles, b_min, b_max, coords, ids = self.load_model()
        system_e = self.compute_total_e(coords, b_min, b_max)
        # initialize box dimension
        box_length = b_max - b_min
        energy_evolution = []
        coord_evolution = []
        # clear old log file
        with open(self.log_filename, "w") as log:
            log.write("")
        accepted_steps = 0

        # Monte Carlo steps
        for step in range(self.n_step): 
            # move a particle
            pid = random.randint(0, n_particles-1)
            old_pos = coords[pid].copy()
            displacement = np.random.uniform(-self.delta, self.delta, size=3)
            new_pos = old_pos + displacement
            new_pos = self.apply_pbc(new_pos, b_min, b_max)
            # compute energy change and calculate acceptance
            dE = self.compute_local_deltaE(coords, pid, old_pos, new_pos, b_min, b_max)
            if dE <= 0 or random.random()<np.exp(-dE/(self.temperature*self.kB)):
                coords[pid] = new_pos
                system_e += dE
                accepted_steps += 1
            # save data at specified intervals
            if step % self.save_interval == 0:
                energy_evolution.append(system_e)
                coord_evolution.append(coords.copy())
                self.save_log(step, system_e, coords, ids)
        
        # Save final step if not already saved
        final_step = self.n_step - 1
        if final_step % self.save_interval != 0:
            energy_evolution.append(system_e)
            coord_evolution.append(coords.copy())
            self.save_log(final_step, system_e, coords, ids)

        # Calculate acceptance rate
        acceptance_rate = accepted_steps / self.n_step * 100
        
        # Prepend summary to log file
        with open(self.log_filename, "r") as log:
            existing = log.read()
        with open(self.log_filename, "w") as log:
            log.write(f"Total steps: {self.n_step}\n")
            log.write(f"Accepted steps: {accepted_steps}\n")
            log.write(f"Acceptance rate: {acceptance_rate:.2f}%\n\n")
            log.write(existing)
        
        print(f"Simulation of {self.n_step} steps for {os.path.splitext(os.path.basename(self.model))[0]} is finished")
        print(f"Acceptance rate: {acceptance_rate:.2f}%")
        return coords, energy_evolution, coord_evolution
    

# -------------------------- test features of class MCSimulation ---------------------
if __name__ == "__main__":
    delta = 3.5
    n_step = 80000
    n_parrallel = 10
    for i in range(n_parrallel):
        sim = MCSimulation(
            model=f"/Volumes/HardDevice/Fall25/CHEME6130/assignment6/mini/8_0.064_{i+1}.txt",
            epsilon=1, sigma=1, cutoff=2.5, n_step=n_step, 
            temperature=2, kB=1, delta=delta, save_interval=50,
            log_filename=f"d{delta}_s{n_step}_{i+1}.txt"
        )
        print(f"Finished parallel simulation {i+1}")
        final_coords, energy_history, coords_history = sim.run_simulation()