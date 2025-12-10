import os
import pickle
import numpy as np
from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=67.7, Om0=0.2592, Tcmb0=2.725)


import DataReader
import Utilities
CS = Utilities.Constants()
DU = Utilities.DataUtilities()

class SinkData:
    def __init__(self, base):
        self.base = base
        self.Reader = DataReader.Reader(self.base)
        self.sink_particles = self.Reader.pickle_reader("sink_particle_reduced.pkl")

        self.ids = np.array(list(self.sink_particles.keys()))
        self.M_init = np.array([self.sink_particles[ID]["meta"]["M_init"] for ID in self.ids])
        self.M_final = np.array([self.sink_particles[ID]["meta"]["M_final"] for ID in self.ids])
        self.M_diff = self.M_final - self.M_init
        self.Type = np.array([self.sink_particles[ID]["meta"]["Type"] for ID in self.ids])
        self.FormationSnap = np.array([self.sink_particles[ID]["meta"]["FormationSnap"] for ID in self.ids])
        self.Status = [self.sink_particles[ID]["meta"]["Status"] for ID in self.ids]
          
    def create_text_file(self):
        files = os.listdir(self.base)
        if "SinkProperties.txt" in files:
            print("File already exists.")
            return
        PickleReader = DataReader.Reader(self.base)
        sink_particles = PickleReader.pickle_reader("sink_particle.pkl")
        sink_particles_reduced = PickleReader.pickle_reader("sink_particle_reduced.pkl")       
        file = f"{self.base}SinkProperties.txt"
        with open(file, "w") as f:
            f.write(f"ID M_init M_final M_diff Type FormationSnap\n")

            sink_id = sink_particles.keys()
            for i, id in enumerate(sink_id):
                entries = sink_particles[id]["evolution"]
                time = np.array(sorted(entries.keys()))
                StellarMass = np.array([entries[t]["StellarMass"] for t in time])
                Type = np.array([entries[t]["Type"] for t in time])
                FS = sink_particles_reduced[id]["meta"]["FormationSnap"]

                f.write(f"{id} {StellarMass[0]} {StellarMass[-1]} {StellarMass[-1] - StellarMass[0]} {Type[0]} {FS} \n")
        f.close()

    def _load_sink_data(self):
        files = os.listdir(self.base)
        if "SinkProperties.txt" not in files:
            self.create_text_file()
        self.ids, self.M_init, self.M_final, self.M_diff, self.Type, self.FormationSnap = np.genfromtxt(f"{self.base}SinkProperties.txt", usecols=(0, 1, 2, 3, 4, 5), skip_header=1, unpack=True)
        
    def filter_by_type(self, t, kind="mask"):
        ### kind = "mask", we just return the mask
        ### kind = "value", we return the values after the mask.
        ### kind = "self", we change the values inside the class for internal use.
        mask = self.Type == t
        if kind == "mask":
            return mask
        elif kind == "value":
            return self.ids[mask], self.M_init[mask], self.M_final[mask], self.M_diff[mask], self.FormationSnap[mask]
        elif kind == "self":
            self.ids, self.M_init, self.M_final, self.M_diff, self.FormationSnap = self.ids[mask], self.M_init[mask], self.M_final[mask], self.M_diff[mask], self.FormationSnap[mask]
    
    def filter_by_accretion(self, threshold, kind="mask"):
        threshold *= CS.hubble_parameter / 1e10
        mask = self.M_diff > threshold
        if kind == "mask":
            return mask
        elif kind == "value":
            return self.ids[mask], self.M_init[mask], self.M_final[mask], self.M_diff[mask], self.FormationSnap[mask]
        elif kind == "self":
            self.ids, self.M_init, self.M_final, self.M_diff, self.FormationSnap = self.ids[mask], self.M_init[mask], self.M_final[mask], self.M_diff[mask], self.FormationSnap[mask]
            return
    
    def max_mass(self, t=0):
        mask = self.Type == t
        return np.max(self.M_final[mask])
    
    def preliminary_statistics(self):
        print(f"There are {len(self.ids)} sink particles in the simulation.")

        mask = self.filter_by_type("PopIII")
        print(f"There are {len(self.ids[mask])} PopIII stars in the simulation.")
        mask = self.filter_by_type("PopII")
        print(f"There are {len(self.ids[mask])} PopII star clusters in the simulation.")
        mask = self.filter_by_type("MBH")
        print(f"There are {len(self.ids[mask])} heavy seeds in the simulation.")

        mask = self.filter_by_accretion(self.M_init * 1e10 / CS.hubble_parameter)

        print(f"There are {len(self.ids[mask])} BHs that doubled their initial mass")

class SinkParticleMethod(SinkData):
    def __init__(self, base):
        super().__init__(base)

    def BlackHoleMassFunction(self):

        self.filter_by_type("MBH", kind="self")
        MBHMass = []#self.M_init * 1e10 / CS.hubble_parameter
        MBHAccretion = []
        MBHIds = self.ids
        for id in self.sink_particles.keys():
            if id not in MBHIds: continue
            evolution = self.sink_particles[id]["evolution"]
            mass = np.array([evolution[t]["StellarMass"] for t in sorted(evolution.keys())]) * 1e10 / CS.hubble_parameter
            if len(mass) < 2: continue
            MBHMass.append(mass[-1])
            if mass[-1] > mass[-2]: MBHAccretion.append(mass[-1])  
        
        log_min = 5 #np.min(np.log10(MBHMass))
        log_max = np.max(np.log10(MBHMass))
        num_bins = 10
        log_bins = np.linspace(log_min, log_max, num_bins+1)
        bin_edges = 10**log_bins
        bin_centers = 10**((log_bins[:-1] + log_bins[1:]) / 2)
        counts1, _ = np.histogram(MBHMass, bins=bin_edges)
        counts2, _ = np.histogram(MBHAccretion, bins=bin_edges)

        box_volume = 1.0 #Mpc^3
        bin_wdith_dex = log_bins[1] - log_bins[0]

        dN_dM1 = counts1 / (box_volume * bin_wdith_dex * 216) 
        dN_dM2 = counts2 / (box_volume * bin_wdith_dex * 216)

        return dN_dM1, dN_dM2, bin_centers


class SinkDataAll:
    def __init__(self, base, feedback=False):
        self.base = base
        self.feedback = feedback
        self.BinaryReader = DataReader.BinaryReader(self.base, self.feedback)
        self.Reader = DataReader.Reader(self.base)
        files = os.listdir(self.base)
        sink_particle_file = "sink_particle_mini.pkl" if "sink_particle_mini.pkl" in files else "sink_particle.pkl"
        if "sink_particle.pkl" in files:
            print(f"Opening pickle file {sink_particle_file}")
            self.sink_particles = self.Reader.pickle_reader(sink_particle_file)
        else:
            self.BinaryReader.read_sink_info()
            self.sink_particles = self.Reader.pickle_reader("sink_particle.pkl")
    
    def create_sink_particles_reduced(self):
        sink_particle_reduced_file = "sink_particle_reduced.pkl"

        sink_id = np.array(list(self.sink_particles.keys()))
        num_sinks = len(sink_id)
        print(f"Number of sink particles in the simulation are {num_sinks}")
        id_file = self.base + "sink_id.txt"
        with open(id_file, "w") as f:
            f.write("SinkID\n")
            for i in range(num_sinks):
                f.write(str(sink_id[i])+ "\n")
        
        start_snap, end_snap = DU.file_range(self.base)
        sink_particle_reduced = {}
        
        for snap in range(start_snap, end_snap):
            self.BinaryReader.read_sink_snap(snap)
            time = self.BinaryReader.time
            SinkID = self.BinaryReader.extract_data("ID")
            SinkMass = self.BinaryReader.extract_data("StellarMass")
            SinkType = self.BinaryReader.extract_data("Type")
            SinkPos = self.BinaryReader.extract_data("Pos")
            SinkVel = self.BinaryReader.extract_data("Vel")

            for i, id in enumerate(SinkID):
                if id not in sink_particle_reduced:
                    if id not in self.sink_particles: continue
                    meta = self.sink_particles[id]["meta"]
                    sink_particle_reduced[id] = {
                        "meta": {
                            "FormationSnap": snap,
                            "M_init": SinkMass[i],
                            "M_final": SinkMass[i] if snap == end_snap - 1 else None,
                            "StellarLifeTime": meta["StellarLifeTime"],
                            "FormationTime": meta["FormationTime"],
                            "Type": meta["Type"], # Can be PopII, PopIII, or MBH
                            "SNeType": meta["SNeType"], # Can be No SNe, TypeII SNe, DCBH, PISN
                            "Status": meta["Status"],
                            
                        },
                        "evolution": {}
                    }
                
                sink_particle_reduced[id]["evolution"][time] = {
                    "Time": time,
                    "Pos": SinkPos[i],
                    "Vel": SinkVel[i],
                    "StellarMass": SinkMass[i],
                    "Type": SinkType[i],
                }
        for id in sink_id:
            if id not in sink_particle_reduced:
                meta = self.sink_particles[id]["meta"]
                evolution = self.sink_particles[id]["evolution"]
                times = np.array(list(sorted(evolution.keys())))
                init_time = np.min(times)
                final_time = np.max(times)
                sink_particle_reduced[id] = {
                    "meta": {
                        "FormationSnap": None,
                        "M_init": evolution[init_time]["StellarMass"],
                        "M_final": evolution[final_time]["StellarMass"],
                        "StellarLifeTime": meta["StellarLifeTime"],
                        "FormationTime": meta["FormationTime"],
                        "Type": meta["Type"], # Can be PopII, PopIII, or MBH
                        "SNeType": meta["SNeType"], # Can be No SNe, TypeII SNe, DCBH, PISN
                        "Status": meta["Status"],
                    },
                    "evolution": {}
                }
            if sink_particle_reduced[id]["meta"]["M_final"] == None:
                evolution = self.sink_particles[id]["evolution"]
                times = np.array(list(sorted(evolution.keys())))
                final_time = np.max(times)
                sink_particle_reduced[id]["meta"]["M_final"] = evolution[final_time]["StellarMass"]
        for sink_id in sink_particle_reduced:
            sink_particle_reduced[sink_id]["evolution"] = dict(sorted(sink_particle_reduced[sink_id]["evolution"].items()))
        with open(self.base + sink_particle_reduced_file, "wb") as f:
            pickle.dump(sink_particle_reduced, f)

    def merger_information(self):
        if os.path.exists(f"{self.base}MergerMap.pkl"):
            MergerMap = self.Reader.pickle_reader("MergerMap.pkl")
            return MergerMap
        MergerDeletedID = []
        MergerRemnantID = []
        MergerMap = {}
        for ID, sink in self.sink_particles.items():
            evolution2 = sink["evolution"]
            times = np.array(list(sorted(evolution2.keys())))
            MergerMass = np.array([evolution2[t]["MergerMass"] for t in times])
            StellarMass = np.array([evolution2[t]["StellarMass"] for t in times])
            times = times[:-1]
            StellarMass = StellarMass[:-1]
            deltas = np.diff(MergerMass)
            ind = np.nonzero(deltas)
            deltas = deltas[ind]
            StellarMass = StellarMass[ind]
            times = times[ind]
            if len(deltas) > 0:
                if np.any(deltas < 0):
                    print(f"Warning: There are negative merger masses for sink particle {ID}.")
                    mask = deltas > 0
                    deltas, StellarMass, times = deltas[mask], StellarMass[mask], times[mask]
                    uniqueind = np.unique(deltas, return_index=True)[1]
                    deltas, StellarMass, times = deltas[uniqueind], StellarMass[uniqueind], times[uniqueind]
                if ID not in MergerMap:
                    MergerMap[ID] = {}
                MergerMap[ID] = {
                    "M2" : deltas,
                    "Time" : times,
                    "M1" : StellarMass,
                    "Mr" : StellarMass + deltas
                }
        file = f"{self.base}MergerMap.pkl"
        with open(file, "wb") as f:
            pickle.dump(MergerMap, f)

        return MergerMap

    def BlackHoleMassFunction(self, box_volume):
        sink_id = np.array(list(self.sink_particles.keys()))
        MBHMass = []
        MBHAccretion = []
        MBHLuminosity = []
        for i, id in enumerate(sink_id):
            if self.sink_particles[id]["meta"]["Type"] != "MBH": continue
            if self.sink_particles[id]["meta"]["Status"] != "In Simulation": continue
            evolution = self.sink_particles[id]["evolution"]
            times = np.array(sorted(evolution.keys()))
            Redshift = 1 / times - 1
            EvolutionTimes = Time = cosmo.age(Redshift).to("Myr").value
            mass = np.array([evolution[t]["StellarMass"] for t in times]) * 1e10 / CS.hubble_parameter
            if len(mass) < 2: continue
            if mass[-1] < 1e5: continue
            MBHMass.append(mass[-1])
            if mass[-1] > mass[-2]: MBHAccretion.append(mass[-1])

            massdiff = (mass[-1] - mass[-2]) * CS.Msun_to_g
            timediff = (EvolutionTimes[-1] - EvolutionTimes[-2]) * 1e6 * CS.yr_to_s
            AccretionRate = massdiff/ timediff

            Luminosity = 0.1 * AccretionRate * pow(CS.c_in_cgs, 2)
            if Luminosity > 1e42:
                MBHLuminosity.append(mass[-1])
        
        log_min = 5.25 #np.min(np.log10(MBHMass))
        log_max = np.max(np.log10(MBHMass))
        bin_width = 0.50
        log_centers = np.arange(log_min, log_max + bin_width, bin_width)
        log_edges = log_centers - bin_width / 2
        log_edges = np.append(log_edges, log_edges[-1] + bin_width)
        bin_edges = 10**log_edges
        bin_centers = 10**log_centers
        counts1, _ = np.histogram(MBHMass, bins=bin_edges)
        counts2, _ = np.histogram(MBHAccretion, bins=bin_edges)
        counts3, _ = np.histogram(MBHLuminosity, bins=bin_edges)
        bin_wdith_dex = bin_width

        dN_dM1 = counts1 / (box_volume * bin_wdith_dex * 216) 
        dN_dM2 = counts2 / (box_volume * bin_wdith_dex * 216)
        dN_dM3 = counts3 / (box_volume * bin_wdith_dex * 216)

        return dN_dM1, dN_dM2, dN_dM3, bin_centers

    def BlackHoleMassFunction2(self, MostMassiveIDs, box_volume):
        MBHMass = []
        MBHAccretion = []
        MBHLuminosity = []
        for i, id in enumerate(MostMassiveIDs):
            if id == -1: continue
            if id not in self.sink_particles: continue
            #if self.sink_particles[id]["meta"]["Type"] != "MBH": continue
            #if self.sink_particles[id]["meta"]["Status"] != "In Simulation": continue
            evolution = self.sink_particles[id]["evolution"]
            times = np.array(sorted(evolution.keys()))
            Redshift = 1 / times - 1
            EvolutionTimes = Time = cosmo.age(Redshift).to("Myr").value
            mass = np.array([evolution[t]["StellarMass"] for t in times]) * 1e10 / CS.hubble_parameter
            if len(mass) < 2: continue
            if mass[-1] < 1e5: continue
            MBHMass.append(mass[-1])
            if mass[-1] > mass[-2]: MBHAccretion.append(mass[-1])

            massdiff = (mass[-1] - mass[-2]) * CS.Msun_to_g
            timediff = (EvolutionTimes[-1] - EvolutionTimes[-2]) * 1e6 * CS.yr_to_s
            AccretionRate = massdiff/ timediff

            Luminosity = 0.1 * AccretionRate * pow(CS.c_in_cgs, 2)
            if Luminosity > 1e42:
                MBHLuminosity.append(mass[-1])
        
        log_min = 5.25 #np.min(np.log10(MBHMass))
        log_max = np.max(np.log10(MBHMass))
        bin_width = 0.50
        log_centers = np.arange(log_min, log_max + bin_width, bin_width)
        log_edges = log_centers - bin_width / 2
        log_edges = np.append(log_edges, log_edges[-1] + bin_width)
        bin_edges = 10**log_edges
        bin_centers = 10**log_centers
        counts1, _ = np.histogram(MBHMass, bins=bin_edges)
        counts2, _ = np.histogram(MBHAccretion, bins=bin_edges)
        counts3, _ = np.histogram(MBHLuminosity, bins=bin_edges)

        bin_wdith_dex = bin_width

        dN_dM1 = counts1 / (box_volume * bin_wdith_dex) 
        dN_dM2 = counts2 / (box_volume * bin_wdith_dex)
        dN_dM3 = counts3 / (box_volume * bin_wdith_dex)

        return dN_dM1, dN_dM2, dN_dM3, bin_centers

    def reduce_sink_particle_file(self):
        sink_particle_mini_file = "sink_particle_mini.pkl"
        sink_particle_mini = {}

        sink_ids = list(self.sink_particles.keys())
        num_sinks = len(sink_ids)
        for sid in sink_ids:
            sink_particle_mini[sid] = {}
            sink_particle_mini[sid]["meta"] = self.sink_particles[sid]["meta"]
            evolution = self.sink_particles[sid]["evolution"]
            times = sorted(evolution.keys())
            times_mini = times[::10]
            sink_particle_mini[sid]["evolution"] = {}
            for t in times_mini:
                sink_particle_mini[sid]["evolution"][t] = evolution[t]
        
        with open(self.base + sink_particle_mini_file, "wb") as f:
            pickle.dump(sink_particle_mini, f)

    def LuminosityFunction(self):
        sink_id = np.array(list(self.sink_particles.keys()))
        MBHMass = []
        Lbol = []
        for i, id in enumerate(sink_id):
            if self.sink_particles[id]["meta"]["Type"] != "MBH": continue
            evolution = self.sink_particles[id]["evolution"]
            times = np.array(sorted(evolution.keys()))
            Redshift = 1 / times - 1
            EvolutionTimes = Time = cosmo.age(Redshift).to("Myr").value
            mass = np.array([evolution[t]["StellarMass"] for t in times]) * 1e10 / CS.hubble_parameter
            if len(mass) < 2: continue
            if mass[-1] < 1e5: continue

            massdiff = (mass[-1] - mass[-2]) * CS.Msun_to_g
            timediff = (EvolutionTimes[-1] - EvolutionTimes[-2]) * 1e6 * CS.yr_to_s
            AccretionRate = massdiff / timediff

            Luminosity = 0.1 * AccretionRate * pow(CS.c_in_cgs, 2)
            if Luminosity > 1e42:
                MBHMass.append(mass[-1])
                Lbol.append(Luminosity)
        
        MBHMass, Lbol = np.array(MBHMass), np.array(Lbol)

        return MBHMass, Lbol
             
