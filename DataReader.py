import numpy as np
import pickle
import Utilities
DU = Utilities.DataUtilities()

class Reader:
    def __init__(self, base):
        self.base = base
    
    def pickle_reader(self, filename):
        pickle_file = f"{self.base}{filename}"
        
        with open(pickle_file, "rb") as f:
            data = pickle.load(f)
        
        return data
    
    def text_reader(self, filename):
        file = f"{self.base}{filename}"
        with open(file, "r") as f:
            header = f.readline().strip("\n").split(" ")
            data = f.readlines()
            dict_data = {h : [] for h in header}
            for i, line in enumerate(data):
                parts = line.split(" ")
                for j, part in enumerate(parts):
                    try:
                        dict_data[header[j]].append(float(part))
                    except ValueError:
                        dict_data[header[j]].append(part)
            
        return dict_data



import struct
import os
class BinaryReader:
    def __init__(self, base, feedback=False, array_length=1):
        self.base = base
        self.feedback = feedback
        if self.feedback:
            main = (
                "3d 3d 3d d d d d q i i i i d d d d d d d d i "  # Scalars and small arrays
                + "200d "  # explosion_time[200]
                + "200d "  # stellar_mass[200]
            )
            if array_length == 1:
                extra = "d d" # MassStillToConvert[350]  AccretionTime[350]
            else:
                extra = "350d 350d"
            self.struct_format = main + extra
        else:
            main = (
                "3d 3d 3d d d d d q i i i i d d d d d d i "  # Scalars and small arrays
                + "200d "  # explosion_time[200]
                + "200d "  # stellar_mass[200]
            )
            if array_length == 1:
                extra = "d d" # MassStillToConvert[350]  AccretionTime[350]
            else:
                extra = "350d 350d"
            self.struct_format = main + extra
            
        self.struct_size = struct.calcsize(self.struct_format)
       
    def read_sink_info(self):
        sink_base = f"{self.base}sink_particle_info/"
        files = os.listdir(sink_base)
        files = sorted(files)
        print(files)

        sink_particles = {}
        times = []

        for i in range(len(files)):
            filename = f"{sink_base}{files[i]}"
            print(f"Opening file {filename}")

            with open(filename, "rb") as f:
                while True:
                    chunk = f.read(8)
                    if not chunk or len(chunk) < 8:
                        break
                    time = struct.unpack("d", chunk)[0]
                    if (time < 0) or (time > 1):
                        break
                    num_sinks = struct.unpack("i", f.read(4))[0]
                    if (num_sinks < 0) or (num_sinks > 50000):
                        break
                    print(f"Time of the dump: {time}. Number of sink particles: {num_sinks}")

                    for _ in range(num_sinks):
                        try:
                            raw = f.read(self.struct_size)
                            if len(raw) < self.struct_size:
                                raise struct.error("Incomplete data.")
                                break
                            data = struct.unpack(self.struct_format, raw)
                        except:
                            print(f"Incomplete sink entry at time {time}, aborting for this timestep.")
                            break
                        times.append(time)
                        sink_id = data[13]
                        entry = {
                            "Time": time,
                            "Pos": [data[0:3]],
                            "Vel": [data[3:6]],
                            "Type": data[17],
                            "StellarMass": data[23] if self.feedback else data[21],
                            "MergerMass": data[24] if self.feedback else data[22],
                        }
                        if entry["Type"] not in [0, 3]: continue
                        if sink_id not in sink_particles:
                            print(f"Adding sink_particle ID {sink_id}")
                            if sink_id < 0:
                                break
                            sink_particles[sink_id] = {
                                "meta": {
                                    "StellarLifeTime": data[19],
                                    "FormationTime": data[12],
                                    "Type": DU.Type(data[17]), # Can be PopII, PopIII, or MBH
                                    "SNeType": DU.SNeType(data[9]) if data[17] == 0 else DU.Type(data[17]), # Can be No SNe, TypeII SNe, DCBH, PISN
                                    "Status": None,
                                    
                                },
                                "evolution": {}
                            }
                        sink_particles[sink_id]["evolution"][time] = entry
        
        for sink_id in sink_particles:
            sink_particles[sink_id]["evolution"] = dict(sorted(sink_particles[sink_id]["evolution"].items()))
        
        final_time = np.max(np.array(times))
        for sink_id in sink_particles:
            meta = sink_particles[sink_id]["meta"]
            evolution = sink_particles[sink_id]["evolution"]
            last_present = final_time in evolution

            if meta["Type"] == "PopII":
                meta["Status"] = "In Simulation"
            elif meta["Type"] == "PopIII":
                if meta["SNeType"] == "PISN":
                    meta["Status"] = "In Simulation" if last_present else "Deleted"
                else:
                    meta["Status"] = "In Simulation" if last_present else "Merged"
            elif meta["Type"] == "MBH":
                meta["Status"] = "In simulation" if last_present else "Merged"

        pickle_file = f"{self.base}sink_particle.pkl"
        with open(pickle_file, "wb") as f:
            pickle.dump(sink_particles, f)
        
        print("sink_particle.pkl file created.")

    def read_sink_snap(self, snap):
        filename = f"{self.base}sink_snap_{snap:03d}"
        with open(filename, "rb") as f:
            time = struct.unpack("d", f.read(8))[0]
            Redshift = 1 / time - 1
            num_sinks = struct.unpack("i", f.read(4))[0]
            print(f"Time of snapshot: {time}. Redshift of the snapshot: {Redshift}. Number of sink particles: {num_sinks}")

            sink_particles = []
            for _ in range(num_sinks):
                data = struct.unpack(self.struct_format, f.read(self.struct_size))

                sink_particles.append({
                    "Pos": data[0:3],
                    "Vel": data[3:6],
                    "Mass": data[9],
                    "ID": data[13],
                    "Type": data[17],
                    "StellarMass" : data[23] if self.feedback else data[21],
                    "MergerMass": data[24] if self.feedback else data[22],
                    "N_SNe": data[26] if self.feedback else data[24]
                })
        self.time = time
        self.num_sinks = num_sinks
        self.sink_particles = sink_particles
    
    def extract_data(self, field):
        return np.array([sink[field] for sink in self.sink_particles])   

    