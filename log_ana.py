# -------------------------- imports and settings ---------------------
import numpy as np
import random
import os 


# -------------------------- load and read data file ---------------------
"""
1. access to trajectory file
    move to given frame (how many frames in total, initial step, final step, thermo length)
        frame ID
        step ID
        total energy
        particle coordinates list
        given pair energy
        single particle potential energy
        RDF

2. access to model file
    particle coordinates list
    total energy
    single particle potential energy
    given pair energy 
    radial distribution function
"""

class Log:
    """log file class that provides analysis features for trajectory information"""
    def __init__(self, path, filename):
        self.path = path
        self.filename = filename


    def load_log(self):
        """load trajectory file(s)"""
        data_file = os.path.join(self.path, self.filename)
        return data_file


    def count_frames(self):
        """count total frame number"""
        data_file = self.load_log()
        f_count = 0
        with open(data_file, "r") as data:
            for line in data:
                if line.startswith("STEP"):
                    f_count += 1
        # print(f"There are {f_count} frames in the {self.filename}")
        return f_count


    def output_frame(self, frame_id):
        """output the content of a frame if given ID (from 0)"""
        data_file = self.load_log()
        f_count = self.count_frames()
        current_frame = -1
        frame_coordinates = []
        frame_info = {}

        with open(data_file, "r") as data:
            lines = data.readlines()
        # avoid unrealistic frame id
        if frame_id > f_count:
            print(f"There are only {f_count} frames")
        frame_info["coordinates"] = frame_coordinates
        # access particular frame if
        i=0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("STEP"):
                current_frame += 1
                if current_frame == frame_id:
                    frame_info["step"] = int(lines[i].strip()[5:]) # output step id
                    frame_info["total energy"] = float(lines[i+1].strip()[6:]) # output total energy
                    j = i + 2
                    while j < len(lines) and not lines[j].startswith("STEP"):
                        frame_coordinates.append(lines[j].strip())
                        j += 1
                    break
            i += 1
        frame_coordinates = frame_coordinates[:-1]
        frame_info["coordinates"] = frame_coordinates
        class Frame:
            """optionally print frame information"""
            def __init__(self, frame_info):
                self.frame_info = frame_info
            def show(self):
                for key, value in self.frame_info.items():
                    print(f"{key} {value}")
        return Frame(frame_info)


    def compute_etotal(self, frame_id):
        """compute total energy"""
        data_file = self.load_log()
        output = self.output_frame(frame_id)
        total_energy = output.frame_info["total energy"]
        # print(total_energy)
        return total_energy


    def compute_pair_energy(self, frame_id, particle1, particle2, epsilon, sigma):
        """compute given pair energy if available"""
        data_file = self.load_log()
        output = self.output_frame(frame_id)
        coords_list = output.frame_info["coordinates"]
        coordinates = np.array([[float(x) for x in coord.split()] for coord in coords_list])

        # remind to undefined particle ID in the box
        if particle1 > len(coordinates) or particle1 <= 0:
            print(f"There are only {len(coordinates)} in the system")
        if particle2 > len(coordinates) or particle2 <= 0:
            print(f"There are only {len(coordinates)} in the system")
        if particle1 == particle2:
            print(f"You indicated the identical particle, no correct definition of pair energy")
        # compute pair energy
        p1_pos = coordinates[particle1-1]
        p2_pos = coordinates[particle2-1]
        r12 = p1_pos - p2_pos
        dist = np.linalg.norm(r12)
        pair_e = 4*epsilon*((sigma/dist)**12 - (sigma/dist)**6)
        print(f"The distance between particle {particle1} and particle {particle2} is {dist:.3f}")
        print(f"The pair energy between particle {particle1} and particle {particle2} is {pair_e:.3f}")


    def compute_local_energy(self, frame_id, particle_id, cutoff, epsilon, sigma):
        """compute total energy of a particle given ID"""
        data_file = self.load_log()
        output = self.output_frame(frame_id)
        coords_list = output.frame_info["coordinates"]
        coordinates = np.array([[float(x) for x in coord.split()] for coord in coords_list])
        n_particles = len(coordinates)

        # find particle list within cutoff and compute local potential
        if particle_id > n_particles or particle_id < 1:
            raise ValueError(f"There are only {len(coordinates)} particles in the system")
        
        target_pos = coordinates[particle_id-1]
        cutoff_list = []
        dists = []
        locel_energy = 0
        for j, pos in enumerate(coordinates):
            if j == particle_id - 1:
                continue
            rij = pos - target_pos
            dist = np.linalg.norm(rij)
            if dist <= cutoff:
                cutoff_list.append(j+1)
                potential = 4*epsilon*((sigma/dist)**12 - (sigma/dist)**6)
                locel_energy += potential
        print(f"Cutoff list of particle {particle_id} is {cutoff_list} that makes local energy {locel_energy}")
        return cutoff_list, locel_energy   


    def compute_rdf(self, density, n_bins, frame_id):
        """compute rdf of a given frame"""
        data_file = self.load_log()
        output = self.output_frame(frame_id)
        coords_list = output.frame_info["coordinates"]
        coordinates = np.array([[float(x) for x in coord.split()] for coord in coords_list])
        n_particle = len(coordinates)
        L = (n_particle*1 / density)**(1/3)
        r_max = L/2
        dr = r_max / n_bins
        bin_particles = np.zeros(n_bins)

        # compute average number of particles in every bin systematically
        for i in range(n_particle):
            for j in range(i+1, n_particle):
                rij = coordinates[i] - coordinates[j]
                rij = rij - L * np.round(rij / L)
                dist = np.linalg.norm(rij)
                bin_id = int(dist/dr)
                if bin_id < n_bins:
                    bin_particles[bin_id] += 2

        # compute system-level radial distribution function
        bins_rdf = []
        for bin_id, bin_particle in enumerate(bin_particles):
            r_id = (bin_id+0.5) * dr
            rdf = bin_particle / ((4*np.pi*r_id**2*dr)*density*n_particle)
            bins_rdf.append(rdf)
        bins_rdf = np.array(bins_rdf)
        #print(bins_rdf)
        return bins_rdf



# -------------------------- test features of class Log ---------------------
if __name__ == "__main__":
    path = "/path/where/model/file/exists/"
    filename = "d0.05_s220000_1.txt"
    log = Log(
        path=path,
        filename=filename
    )
    #log.count_frames()
    #log.output_frame(0).show()
    #log.compute_etotal(0)
    #log.compute_pair_energy(frame_id=0, particle1=2, particle2=1, epsilon=1, sigma=1)
    #log.compute_local_energy(frame_id=0, particle_id=4, cutoff=2.5, epsilon=1, sigma=1)
    log.compute_rdf(density=0.296, n_bins=300, frame_id=399)





            
