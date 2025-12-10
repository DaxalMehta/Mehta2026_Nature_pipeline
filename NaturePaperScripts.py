import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import yt
from mpi4py import MPI
import pickle
import os
import glob
import h5py
from scipy.spatial import cKDTree
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=67.7, Om0=0.2592, Tcmb0=2.725)


import YT
import SinkParticles
import Utilities
import Plotter
import DataReader
import GroupsReader

CS = Utilities.Constants()
DU = Utilities.DataUtilities()

color_schemes = {
    "Okabe–Ito": ['#0072B2', '#E69F00', '#009E73', '#D55E00'],
    "Tol Muted": ['#88CCEE', '#DDCC77', '#117733', '#CC6677'],
    "Material": ['#673AB7', '#FFC107', '#009688', '#E53935'],
    "Categorical": ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728'],
    "ColorBlind" : ['#A00000', '#0072B2', '#3C0550',  '#E69F00'],
}

class Scripts:
    def __init__(self, base):
        self.base = base
        self.output_dir = "NatureManuscriptPlots/"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.dir = ["Level13/", "Level14/", "Level15/", "Level15_feedback/"]
        self.label = ["L13", "L14", "L15", "L15_BHFB"]
        self.feedback = [False, False, False, True]
        self.colors = color_schemes["ColorBlind"]
        self.PickleReader = DataReader.Reader(self.base)
        
    def SinkHistogram(self):
        """
        Generate the mass distribution of PopIII stars for all four simulation channels.

        Produces:
        1. A histogram figure (Mehta_ED_Fig1.jpg)
        2. A source-data TXT file containing the mass arrays for each channel
        """
        print("Entering SinkHistogram() function.")

        # Creating a plot figure
        plot = Plotter.Plotter()
        all_data = []
        names = []
        # Loop over the four simulation setups
        for i in range(4):
            # Load sink data for this simulation channel
            self.BinaryReader = DataReader.BinaryReader(f"{self.base}{self.dir[i]}", feedback=self.feedback[i], array_length=350)
            SinkData = SinkParticles.SinkData(f"{self.base}{self.dir[i]}")
            
            # Filtering out data for just PopIII particles
            SinkData.filter_by_type("PopIII", kind="self")
            MInit = SinkData.M_init * 1e10 / CS.hubble_parameter
            mask = MInit < 300
            MInit = MInit[mask]
            all_data.append(MInit)
            
            # Plotting the data for all setups
            
            bins = DU.histogram_bins(MInit, num=40)
            plot.ax.hist(MInit, bins=bins, color=self.colors[i], histtype="step", lw=1.5, density=True, alpha=0.75, label=self.label[i])
            data_out = np.column_stack((MInit,))
        
        #Figure Formatting
        plot.add_details(plot.ax, xlabel="M$_\mathrm{i} [\mathrm{M}_\odot$]", ylabel="Probability Density", xscale="log", legend=True, grid=True)
        plot.save(f"{self.output_dir}Mehta_ED_Fig1.jpg")
        
        # --- Source data output ---
        # Pad arrays to equal length for simple column-wise saving

        max_len = max(len(arr) for arr in all_data)
        padded = [np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan) for arr in all_data]
        combined = np.column_stack(padded)
        header = "MInit_channel0  MInit_channel1  MInit_channel2  MInit_channel3"
        np.savetxt(f"{self.output_dir}Mehta_SourceData_ED_Fig1.txt",
                combined,
                header=header,
                comments='')
         
    def SinkGrowth(self):
        """
        Generate the mass growth histories for all PopIII particles that accreted more than 0.1 solar mass
        
        Produces: 
        1. Growth histories of BHs particles in redshift. (Mehta_Fig1.pdf)
        2. Growth histories from for BHs particles in time. (Mehta_ED_Fig5.jpg)
        3. Eddington Rate of the most accreting BHs with redshift. (Mehta_Fig3.pdf)
        4. Source data for all three plots. (Mehta_SourceData_Fig1.txt, Mehta_SourceData_ED_Fig5.txt, Mehta_SourceData_Fig3.txt)
        """
        print("Entering the SinkGrowth() function.")
        colors = color_schemes["ColorBlind"]
        plot = Plotter.Plotter(figsize=(9, 6))
        ax2 = plot.ax.twiny()
        plotBH = Plotter.Plotter(figsize=(9, 6))
        plotAR = Plotter.Plotter(figsize=(9, 6))
        ax2AR = plotAR.ax.twiny()
        plotER = Plotter.Plotter(figsize=(9, 6))
        ax2ER = plotER.ax.twiny()
        fname1 = f"{self.output_dir}Mehta_SourceData_Fig1.txt"
        fname2 = f"{self.output_dir}Mehta_SourceData_ED_Fig5.txt"
        fname3 = f"{self.output_dir}Mehta_SourceData_Fig3.txt"
        f = open(fname1, "w")
        f.write("# Source data for SinkGrowth.pdf\n")
        f.write("# Columns: DirIndex\tSinkID\tRedshift\tStellarMass[M_sun]\n")

        ff = open(fname2, "w")
        ff.write("# Source data for Mehta_ED_Fig5.jpg")
        ff.write("# Columns: DirIndex\tSinkID\tTime[Myr]\tBlackHoleMass[M_sun]\n")

        fff = open(fname3, "w")
        fff.write("# Source data for EddingtonRate.pdf")
        fff.write("# Columns: DirIndex\tSinkID\tRedshift\tEddintonRate\n")

        # Looping over all four simulations
        for i in range(4):
            #Loading all sink particle data from the binary files.
            SinkDataAll = SinkParticles.SinkDataAll(f"{self.base}{self.dir[i]}", feedback=self.feedback[i])
            sink_particles = SinkDataAll.sink_particles

            IDS = sink_particles.keys()

            # Looping over all sink_particles. Only plotting for BHs formed from PopIII particles
            for j, ID in enumerate(IDS):
                print(f"\rProcessing sink_particle {j+1}/{len(sink_particles)}", end="")
                meta = sink_particles[ID]["meta"]
                if meta["Type"] == "PopII" or meta["Type"] == "MBH": continue
                evolution = sink_particles[ID]["evolution"]
                times = np.array(list(sorted(evolution.keys())))
                if len(times) < 2: continue

                # Extracting data from the binary files.
                StellarMass = np.array([evolution[t]["StellarMass"] for t in times]) * 1e10 / CS.hubble_parameter
                Type = np.array([evolution[t]["Type"] for t in times])
                mask = Type == 3
                Redshift = 1 / times - 1
                
                if StellarMass[-1] - StellarMass[0] < 0.1: continue
                zorder, alpha, linestyle, lw = 1, 0.2, "--", 2
                if (StellarMass[-1] - StellarMass[0]) > StellarMass[0]: zorder, alpha, linestyle, lw = 2, 1, "-", 2

                # Saving data in txt file.
                for z, m in zip(Redshift, StellarMass):
                    f.write(f"{i}\t{ID}\t{z:.6e}\t{m:.6e}\n")

                plot.ax.plot(Redshift, StellarMass, color=colors[i], alpha=alpha, linestyle=linestyle, lw=lw, zorder=zorder)

                # Calculting Cosmic time from redshift and calculating accretion rates and eddington rates.
                EvolutionTimes = cosmo.age(Redshift).to("Myr").value
                AccretionRate = np.diff(StellarMass) / np.diff(EvolutionTimes * 1e6)
                EvolutionTimes = EvolutionTimes[mask]
                if i == 0 and j+1 == 807: print(EvolutionTimes)
                try:
                    EvolutionTimes -= EvolutionTimes[0]
                except:
                    print(EvolutionTimes)
                
                plotBH.ax.plot(EvolutionTimes, StellarMass[mask], color=colors[i], alpha=alpha, linestyle=linestyle, lw=lw, zorder=zorder)

                for t, m in zip(EvolutionTimes, StellarMass[mask]):
                    ff.write(f"{i}\t{ID}\t{t:.6e}\t{m:.6e}\n")
                if linestyle == "--": continue
                EddingtonLimit = DU.EddingtonRate(StellarMass[:-1])
                EddingtonRate = AccretionRate / EddingtonLimit

                maskAR = AccretionRate != 0
                maskER = EddingtonRate != 0
                plotAR.ax.plot(Redshift[:-1][maskAR], AccretionRate[maskAR], color=colors[i], alpha=alpha, linestyle=linestyle, lw=lw, zorder=zorder)
                plotER.ax.plot(Redshift[:-1][maskER], EddingtonRate[maskER], color=colors[i], alpha=alpha, linestyle=linestyle, lw=lw, zorder=zorder)

                for z, val in zip(Redshift[:-1][maskER], EddingtonRate[maskER]):
                    fff.write(f"{i}\t{ID}\t{z:.6e}\t{val:.6e}\n")
        
        # Configuring the plots
        plot.ax.invert_xaxis()
        RedshiftTicks = np.linspace(plot.ax.get_xlim()[0], plot.ax.get_xlim()[1], 6)
        TimeTicks = cosmo.age(RedshiftTicks).to("Myr").value
        ax2.set_xlim(plot.ax.get_xlim())
        ax2.set_xticks(RedshiftTicks)
        ax2.set_xticklabels([f"{t:.1f}" for t in TimeTicks])
        ax2.set_xlabel("Time [Myr]", fontsize=15)
        plot.add_legends(color=colors[0], label="L13")
        plot.add_legends(color=colors[1], label="L14")
        plot.add_legends(color=colors[2], label="L15")
        plot.add_legends(color=colors[3], label="L15_BHFB")
        plot.add_legends(color="black", label="$\Delta \mathrm{M} > \mathrm{M}_{\mathrm{i}}$")
        plot.add_legends(color="black", label="$\Delta \mathrm{M} > 0.1 \mathrm{M}_\odot$", linestyle="--")
        plot.add_details(plot.ax, xlabel="Redshift", ylabel="Mass [M$_\odot$]", yscale="log", xlim=(33, 18))
        plot.save(f"{self.output_dir}Mehta_Fig1.pdf")
        
        plotBH.add_legends(color=colors[0], label="L13")
        plotBH.add_legends(color=colors[1], label="L14")
        plotBH.add_legends(color=colors[2], label="L15")
        plotBH.add_legends(color=colors[3], label="L15_BHFB")
        plotBH.add_legends(color="black", label="$\Delta \mathrm{M} > \mathrm{M}_{\mathrm{i}}$")
        plotBH.add_legends(color="black", label="$\Delta \mathrm{M} > 0.1 \mathrm{M}_\odot$", linestyle="--")
        plotBH.add_details(plotBH.ax, xlabel="Time [Myr]", ylabel="Black Hole Mass [M$_\odot$]", yscale="log", xscale="log", xlim=(1e-1, 1e2))
        plotBH.save(f"{self.output_dir}Mehta_ED_Fig5.jpg")

        plotAR.ax.invert_xaxis()
        RedshiftTicks = np.linspace(plotAR.ax.get_xlim()[0], plotAR.ax.get_xlim()[1], 6)
        TimeTicks = cosmo.age(RedshiftTicks).to("Myr").value
        ax2AR.set_xlim(plotAR.ax.get_xlim())
        ax2AR.set_xticks(RedshiftTicks)
        ax2AR.set_xticklabels([f"{t:.1f}" for t in TimeTicks])
        ax2AR.set_xlabel("Time [Myr]", fontsize=15)
        plotAR.add_legends(color=colors[0], label="L13")
        plotAR.add_legends(color=colors[1], label="L14")
        plotAR.add_legends(color=colors[2], label="L15")
        plotAR.add_legends(color=colors[3], label="L15_BHFB")
        plotAR.add_details(plotAR.ax, xlabel="Redshift", ylabel="Accretion Rate [M$_\odot / yr$]", yscale="log", ylim=(1e-5, 1e2))
        plotAR.save(f"{self.output_dir}AccretionRate.jpg")

        plotER.ax.invert_xaxis()
        RedshiftTicks = np.linspace(plotER.ax.get_xlim()[0], plotER.ax.get_xlim()[1], 6)
        TimeTicks = cosmo.age(RedshiftTicks).to("Myr").value
        ax2ER.set_xlim(plotER.ax.get_xlim())
        ax2ER.set_xticks(RedshiftTicks)
        ax2ER.set_xticklabels([f"{t:.1f}" for t in TimeTicks])
        ax2ER.set_xlabel("Time [Myr]", fontsize=15)
        plotER.add_legends(color=colors[0], label="L13")
        plotER.add_legends(color=colors[1], label="L14")
        plotER.add_legends(color=colors[2], label="L15")
        plotER.add_legends(color=colors[3], label="L15_BHFB")
        plotER.add_details(plotER.ax, xlabel="Redshift", ylabel="f$_{\mathrm{edd}}$", yscale="log", ylim=(1e-5, 1e4), xlim=(33, 18))
        plotER.save(f"{self.output_dir}Mehta_Fig3.pdf", legend_loc="upper right")

        f.close()
        ff.close()
        fff.close()

    def SurroundingGasProperties(self):
        """
        Generates a plot for properties of gas surrounding the growing BHs.

        Produces:
        1. A plot of gas properties evolution along with redshift. (Mehta_Fig4.pdf)
        2. A plot of average metallicity of gas for the BH duty cycle. (Mehta_ED_Fig6.jpg)
        3. Source data txt files for both plots. (Mehta_SourceData_Fig4, Mehta_SourceData_ED_Fig6.txt) 
        """
        print("Entering the SurroundinGasProperties function.")
        colors = color_schemes["ColorBlind"]

        source_data = []
        metallicity_data = []
        plot = Plotter.Plotter(figsize=(10, 10), xsize=4, ysize=1)
        plotMetallicity = Plotter.Plotter()
        
        # Looping over the four simulations
        for i in range(4):
            start_snap, end_snap = DU.file_range(f"{self.base}{self.dir[i]}")
            # Loading in basic data for BHs that accreted more than 0.1 solar masses
            id, snaps, m_init, m_final, m_diff = np.genfromtxt(f"{self.dir[i]}sink_0.1.txt", usecols=(0,1,2,3,4), skip_header=1, unpack=True)
            first_snap = np.min(snaps).astype(int)

            SinkData = SinkParticles.SinkData(f"{self.base}{self.dir[i]}")
            sink_particles = SinkData.sink_particles

            # Loading the data of gas surrouning the BHs (Processed from larger datasets)
            MassSink = np.load(f"{self.dir[i]}mass_sink.npy")
            MassInfalling = np.load(f"{self.dir[i]}mass_infalling.npy")
            MassSurrounding = np.load(f"{self.dir[i]}mass_surrounding.npy")
            Mdot = np.load(f"{self.dir[i]}mdot.npy")
            Redshift = np.load(f"{self.dir[i]}redshift.npy")
            Velocity = np.load(f"{self.dir[i]}velocity.npy")
            Temperature = np.load(f"{self.dir[i]}Temperature.npy")
            Metallicity = np.load(f"{self.dir[i]}Metallicity.npy")
            ZAvg = np.zeros(len(id), float)

            # Looping over the BHs
            for j in range(len(id)):
                if m_init[j] > 300: continue

                mass_row = MassSink[j]
                zero_locs = np.where(mass_row == 0)[0]
                ind_i = zero_locs[-1] if zero_locs.size > 0 else 0

                max_val = mass_row.max()
                max_locs = np.where(mass_row == max_val)[0]
                ind_f = max_locs[0] - 1

                print(f"Starting index: {ind_i}, final index: {ind_f}")

                z_slice = Metallicity[j, ind_i:ind_f]

                mask = z_slice != 0
                ZAvg[j] = np.mean(z_slice[mask]) if np.any(mask) else 0.0
                
                if np.sum(MassSurrounding[j, :]) > 1e8: continue

                evolution = sink_particles[id[j]]["evolution"]
                times = evolution.keys()
                Type = np.array([evolution[t]["Type"] for t in times])
                print(Type, first_snap)
                Type = Type[Type == 3.0]
                index = len(Redshift) - len(Type)

                if m_diff[j] > m_init[j]: 
                    zorder, alpha, linestyle = 3, 0.8, "-"
                    category = 1
                else: 
                    zorder, alpha, linestyle = 1, 0.2, "--"
                    category = 2

                plot.ax[0].plot(Redshift[index:], MassInfalling[j, index:], color=colors[i], alpha=alpha, linestyle=linestyle, zorder=zorder)
                plot.ax[1].plot(Redshift[index:], Temperature[j, index:], color=colors[i], alpha=alpha, linestyle=linestyle, zorder=zorder)
                plot.ax[2].plot(Redshift[index:], Velocity[j, index:], color=colors[i], alpha=alpha, linestyle=linestyle, zorder=zorder)
                plot.ax[3].plot(Redshift[index:], Mdot[j, index:], color=colors[i], alpha=alpha, linestyle=linestyle, zorder=zorder)

                for kk in range(index, len(Redshift)):
                    source_data.append({
                        "dataset": i,
                        "id": float(id[j]),
                        "z": float(Redshift[kk]),
                        "Mass10pc": float(MassInfalling[j, kk]),
                        "Temperature": float(Temperature[j, kk]),
                        "Velocity": float(Velocity[j, kk]),
                        "Mdot": float(Mdot[j, kk]),
                        "category": category

                    })
                metallicity_data.append({
                    "dataset": i,
                    "id": float(id[j]),
                    "Zavg": float(ZAvg[j]),
                    "M_final": float(m_final[j])
                })
            plotMetallicity.ax.scatter(ZAvg, m_final, color=colors[i])
        print(f"Done for {self.dir[i]}", flush=True)

        # Configuring the plots       
        for k in range(4):  
            plot.ax[k].invert_xaxis()
        for k in range(3):
            plot.ax[k].tick_params(bottom=False, labelbottom=False)
        plot.add_details(plot.ax[0], yscale="log", ylabel="$\mathrm{M}_{\mathrm{10pc}} [\mathrm{M}_\odot]$", minorticks=False, xlim=(33, 15))
        plot.add_details(plot.ax[1], yscale="log", ylabel=" T [K]", minorticks=False, xlim=(33, 15))
        plot.add_details(plot.ax[2], yscale="log", ylabel="$\mathrm{V}_{\mathrm{10pc}}$ [km/s]", minorticks=False, xlim=(33, 15))
        plot.add_details(plot.ax[3], yscale="log", xlabel="z", ylabel="$\dot{\mathrm{M}}_{\mathrm{BH}} [\mathrm{M}_\odot/\mathrm{yr}]$", ylim=(1e-9, 4e1), minorticks=False, xlim=(33, 15))

        legend_elements1 = [
        Line2D([0], [0], color=self.colors[0], label=self.label[0], markerfacecolor=self.colors[0], markersize=4, linestyle="-"),
        Line2D([0], [0], color=self.colors[1], label=self.label[1], markerfacecolor=self.colors[1], markersize=4, linestyle="-"),
        Line2D([0], [0], color=self.colors[2], label=self.label[2], markerfacecolor=self.colors[2], markersize=4, linestyle="-"),
        Line2D([0], [0], color=self.colors[3], label=self.label[3], markerfacecolor=self.colors[3], markersize=4, linestyle="-")     
        ]
        legend_elements2 = [
            Line2D([0], [0], color="black", label="$\Delta \mathrm{M} > \mathrm{M}_{\mathrm{i}}$", linestyle="-", markersize=4),
            Line2D([0], [0], color="black", label="$\Delta \mathrm{M} > 0.1 \mathrm{M}_\odot$", linestyle="--", markersize=4)
        ]
        plot.ax[0].legend(handles=legend_elements1, loc="upper left", frameon=False)
        plot.ax[1].legend(handles=legend_elements2, loc="upper left", frameon=False)
        
        texts = ["a", "b", "c", "d"]

        for k in range(4):
            plot.ax[k].text(0.05, 0.15, f"{texts[k]}", transform=plot.ax[k].transAxes, fontsize=15, fontweight="bold", va="top", ha="right")
        
        plot.save(f"{self.output_dir}Mehta_Fig4.pdf")

        plotMetallicity.add_details(plotMetallicity.ax, xscale="log", yscale="log", xlabel="Z [$\mathrm{Z}_\odot$]", ylabel="M$_\mathrm{BH,f} [\mathrm{M}_\odot]$", xlim=(7e-11, 7e-2))

        plotMetallicity.add_legends(type="scatter", color=colors[0], label=self.label[0])
        plotMetallicity.add_legends(type="scatter", color=colors[1], label=self.label[1])
        plotMetallicity.add_legends(type="scatter", color=colors[2], label=self.label[2])
        plotMetallicity.add_legends(type="scatter", color=colors[3], label=self.label[3])

        plotMetallicity.save(f"{self.output_dir}Mehta_ED_Fig6.jpg", legend_loc="upper right")

        # Saving the plotting data in txt files.
        outfile = f"{self.output_dir}Mehta_SourceData_Fig4.txt"
        with open(outfile, "w") as f:
            for item in source_data:
                f.write(json.dumps(item)+"\n")
        print(f"Saved Source data at {outfile}")

        outfile2 = f"{self.output_dir}Mehta_SourceData_ED_Fig6.txt"
        with open(outfile2, "w") as f:
            for item in metallicity_data:
                f.write(json.dumps(item)+"\n")
        print(f"Saved Source data at {outfile2}")

    def HostHaloProperties(self):
        """
        Generates a plot to correlate the BH growth to its host halo properties.
        
        Produces:
        1. A plot of BH final masses as a function of its host halo properties. (Mehta_ED_Fig7.jpg)
        2. Source data txt for the plot. (Mehta_SourceData_ED_Fig7.txt)
        """
        print("Entering HostHaloProperties function.")
        plot = Plotter.Plotter(figsize=(9, 5), xsize=1, ysize=2)

        output = {
            "PANEL_A_DATA": [],
            "PANEL_B_DATA": []
        }
        # Looping over the four simulations
        for i in range(4):
            start_snap, end_snap = DU.file_range(f"{self.base}{self.dir[i]}")
            # Loading the binary file data for sink particles
            self.BinaryReader = DataReader.BinaryReader(f"{self.base}{self.dir[i]}", feedback=self.feedback[i], array_length=350)
            # Loading in the statistical BH data for all BHs that accreted more than 0.1 solar masses
            id, snaps, m_init, m_final, m_diff = np.genfromtxt(f"{self.dir[i]}sink_0.1.txt", usecols=(0,1,2,3,4), skip_header=1, unpack=True)
            first_snap = np.min(snaps).astype(int)

            HaloNrMass = np.zeros(len(id), float)
            HaloNrDensity = np.zeros(len(id), float)
            HaloStellarMass = np.zeros(len(id), float)
            
            for snap in range(first_snap, end_snap):
                if i == 0 and snap == 32: continue

                # Reading sink data for a snapshot
                self.BinaryReader.read_sink_snap(snap)
                if self.BinaryReader.num_sinks == 0: continue
                time = self.BinaryReader.time
                
                id5 = self.BinaryReader.extract_data("ID")
                pos5 = self.BinaryReader.extract_data("Pos")

                #Reading halos and subhalos data for a snapshot
                Groups = GroupsReader.Reader(f"{self.base}{self.dir[i]}", snap)
                halos = Groups.halos
                HaloMass = halos["GroupMass"] * 1e10 / CS.hubble_parameter
                HaloRad = halos["Group_R_TopHat200"] * 1e3 * time / CS.hubble_parameter
                HaloMassType = halos["GroupMassType"] * 1e10 / CS.hubble_parameter 

                subhalos = Groups.subhalos
                SubhaloPos = subhalos["SubhaloCM"]
                SubhaloGrNr = subhalos["SubhaloGroupNr"]
                
                #Finding Host halo properties
                tree = cKDTree(SubhaloPos)
                for j in range(len(id)):
                    if snaps[j] != snap: continue
                    if m_init[j] > 300: continue

                    ind = id5 == id[j]
                    p5 = pos5[ind]
                    _, b = tree.query(p5, k=4)
                    print(b)
                    b = b[0]
                    GroupNr = SubhaloGrNr[b[0]]
                    HaloNrMass[j] = HaloMass[GroupNr]
                    print(GroupNr, HaloNrMass[j])
                    HaloNrDensity[j] = HaloMass[GroupNr] / HaloRad[GroupNr] ** 2
                    HaloStellarMass[j] = HaloMassType[GroupNr, 5]
            
            ind = HaloNrMass > 1e5
            HaloNrMass = HaloNrMass[ind]
            m_diff = m_diff[ind]
            m_init = m_init[ind]
            m_final = m_final[ind]
            mask = m_diff > m_init
            mask2 = np.where((m_diff > 0.1) & (m_diff < m_init) & (m_final < 1e3))[0]
            print(m_init, m_diff)
            print(HaloNrMass[mask], HaloNrMass[mask2])
            plot.ax[0].scatter(HaloNrMass[mask], m_final[mask], color=self.colors[i], alpha=1.0, marker="x", s=30)
            plot.ax[0].scatter(HaloNrMass[mask2], m_final[mask2], color=self.colors[i], alpha=0.3, marker="o", s=20)

            for hm, mf in zip(HaloNrMass[mask], m_final[mask]):
                output["PANEL_A_DATA"].append({
                    "dataset_index": i,
                    "halo_mass": float(hm),
                    "m_final": float(mf),
                    "category": 1
                })
            for hm, mf in zip(HaloNrMass[mask2], m_final[mask2]):
                output["PANEL_A_DATA"].append({
                    "dataset_index": i,
                    "halo_mass": float(hm),
                    "m_final": float(mf),
                    "category": 2
                })
        # Configuring the plots
        plot.add_details(plot.ax[0], xscale="log", yscale="log", xlabel="Halo Mass $[\mathrm{M}_\odot]$", ylabel="M$_\mathrm{f} [\mathrm{M}_\odot]$", ylim=(10, 50000))

        legend_elements = [
            Line2D([0], [0], marker="x", color="black", label="$\Delta \mathrm{M} > \mathrm{M}_{\mathrm{i}}$", linestyle="None", markersize=5),
            Line2D([0], [0], marker="o", color="black", label="$\Delta \mathrm{M} > 0.1 \mathrm{M}_\odot$", linestyle="None", markersize=5)
        ]
        plot.ax[0].legend(handles=legend_elements, loc="upper left", frameon=False, fontsize=11)
        
        for i in range(4):
            start_snap, end_snap = DU.file_range(f"{self.base}{self.dir[i]}")
            id, snaps, m_init, m_final, m_diff = np.genfromtxt(f"{self.dir[i]}sink_0.1.txt", usecols=(0,1,2,3,4), skip_header=1, unpack=True)

            first_snap = np.min(snaps).astype(int)
        
            MassSink = np.load(f"{self.dir[i]}mass_sink.npy")
            MassSurrounding = np.load(f"{self.dir[i]}mass_surrounding.npy")
            Redshift = np.load(f"{self.dir[i]}redshift.npy")
            FormationSigma = np.zeros(len(id))
            for j, ID in enumerate(id):
                if m_init[j] > 300: continue
                if np.sum(MassSurrounding[j, :]) > 1e8: continue

                mass = MassSurrounding[j, int(snaps[j] - first_snap)]
                FormationSigma[j] = mass / (np.pi * 10**2)
            
            mask1 = np.where(m_diff > m_init)[0]
            mask2 = np.where((m_diff > 0.1) & (m_diff < m_init))[0]
            plot.ax[1].scatter(FormationSigma[mask1], m_final[mask1], marker="x", s=20, alpha=1.0, color=self.colors[i], zorder=3)
            plot.ax[1].scatter(FormationSigma[mask2], m_final[mask2], marker="o", s=15, alpha=0.3, color=self.colors[i], zorder=2)

            for sig, mf in zip(FormationSigma[mask1], m_final[mask1]):
                output["PANEL_B_DATA"].append({
                    "dataset_index": i,
                    "sigma": float(sig),
                    "m_final": float(mf),
                    "category": 1
                })
            for sig, mf in zip(FormationSigma[mask2], m_final[mask2]):
                output["PANEL_B_DATA"].append({
                    "dataset_index": i,
                    "sigma": float(sig),
                    "m_final": float(mf),
                    "category": 2
                })
  
        plot.ax[1].tick_params(left=False, labelleft=False)
        texts = ["a", "b"]
        for k in range(2):
            plot.ax[k].text(0.1, 0.1, f"{texts[k]}", transform=plot.ax[k].transAxes, fontsize=12, fontweight="bold", va="top", ha="right")
        legend_elements = [
            Line2D([0], [0], marker="x", color=self.colors[0], label=self.label[0], markerfacecolor=self.colors[0], markersize=5, linestyle="None"),
            Line2D([0], [0], marker="x", color=self.colors[1], label=self.label[1], markerfacecolor=self.colors[1], markersize=5, linestyle="None"),
            Line2D([0], [0], marker="x", color=self.colors[2], label=self.label[2], markerfacecolor=self.colors[2], markersize=5, linestyle="None"),
            Line2D([0], [0], marker="x", color=self.colors[3], label=self.label[3], markerfacecolor=self.colors[3], markersize=5, linestyle="None"),   
        ]
        plot.ax[1].legend(handles=legend_elements, loc="upper left", frameon=False, fontsize=11)

        plot.add_details(plot.ax[1], xscale="log", yscale="log", xlabel="Surface Density $[\mathrm{M}_\odot / \mathrm{pc}^2]$", xlim=(9, 150), ylim=(10, 50000))

        plot.save(f"{self.output_dir}Mehta_ED_Fig7.jpg", w_pad=0.0)

        #----------------------
        #  SOURCE DATA EXPORT
        # ---------------------

        with open(f"{self.output_dir}Mehta_SourceData_ED_Fig7.txt", "w") as f:
            f.write("# HostHaloProperties – Source Data\n\n")
            f.write("PANEL_A_DATA:\n")
            for row in output["PANEL_A_DATA"]:
                f.write(f"{row}\n")

            f.write("\nPANEL_B_DATA:\n")
            for row in output["PANEL_B_DATA"]:
                f.write(f"{row}\n")

    def HMF(self):
        """
        Generates a population of halos and galaxies for all of our simulation.

        Produces.:
        1. A histogram figure. (Mehta_ED_Fig2.jpg)
        2. Source data txt. (Mehta_SourceData_ED_Fig2.txt)
        """
        print("Entering the HMF() function.")

        plot = Plotter.Plotter()
        bins = np.logspace(np.log10(1e5), np.log10(2e9), 100)
        # Storage for source-data
        HaloMass_all = {}
        HaloMass_SF = {}
        
        # Looping over the four simulations.
        for i in range(4):
            start_snap, end_snap = DU.file_range(f"{self.base}{self.dir[i]}")
            
            # Loading the halos and subhalos data
            Groups = GroupsReader.Reader(f"{self.base}{self.dir[i]}", end_snap-1)
            halos = Groups.halos

            HaloMass = halos["GroupMass"] * 1e10 / CS.hubble_parameter
            SinkMass = halos["GroupMassType"][:, 5] * 1e10 / CS.hubble_parameter

            mask = HaloMass > 1e5
            HaloMass = HaloMass[mask]
            SinkMass = SinkMass[mask]
            print(f"Number of Halos in simulation {self.dir[i]} is {len(HaloMass)}.")

            HaloMass_all[i] = HaloMass
            
            plot.ax.hist(HaloMass, bins=bins, histtype="step", color=self.colors[i], zorder=1, linestyle="--")

            mask = SinkMass > 0
            HaloMass = HaloMass[mask]

            HaloMass_SF[i] = HaloMass

            print(f"Number of halos with star formation is {len(HaloMass)}.")

            plot.ax.hist(HaloMass, bins=bins, histtype="step", color=self.colors[i], zorder=2, lw=1.5)
        
        # Plot configuration and legends.
        plot.add_details(plot.ax, xlabel="Halo Mass [M$_\odot$]", ylabel="Number", yscale="log", xscale="log", grid=False)
        plot.add_legends(type="patch", color=self.colors[0], label=f"{self.label[0]} (z = 14.5)", lw=1.5)
        plot.add_legends(type="patch", color=self.colors[1], label=f"{self.label[1]} (z = 18.8)", lw=1.5)
        plot.add_legends(type="patch", color=self.colors[2], label=f"{self.label[2]} (z = 21.0)", lw=1.5)
        plot.add_legends(type="patch", color=self.colors[3], label=f"{self.label[3]} (z = 21.0)", lw=1.5)
        plot.add_legends(type="patch", color="black", label=f"Halos", linestyle="--", lw=1.5)
        plot.add_legends(type="patch", color="black", label=f"Galaxies", lw=1.5)

        plot.save(f"{self.output_dir}Mehta_ED_Fig2.jpg", legend_loc="upper right")

        #----------------------
        #  SOURCE DATA EXPORT
        # ---------------------

        print("Writing Source Data for ED Fig 2")
        
        all_arrays = []
        names = []

        for i in range(4):
            all_arrays.append(HaloMass_all[i])
            names.append(f"AllHalos_sim{i}")

        for i in range(4):
            all_arrays.append(HaloMass_SF[i])
            names.append(f"SFHalos_sim{i}")

        # equal-length padding
        max_len = max(len(arr) for arr in all_arrays)
        padded = [np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan)
                for arr in all_arrays]

        # include bin edges too (as required for histograms)
        bin_pad = np.pad(bins, (0, max_len - len(bins)), constant_values=np.nan)

        combined = np.column_stack([bin_pad] + padded)

        header = "BinEdges " + " ".join(names)

        np.savetxt(f"{self.output_dir}Mehta_SourceData_ED_Fig2.txt",
                combined,
                header=header,
                comments='')

        print("Mehta_SourceData_ED_Fig2.txt written successfully.")

    def PopIIIBHRelation(self):
        """
        Generates a plot for the final masses of the BHs as a function of their progenitor PopIII mass

        Produces:
        1. A scatter plot. (Mehta_ED_Fig3.jpg)
        2. Source data txt. (Mehta_SourceData_ED_Fig3.txt)
        """
        print("Entering PopIIIBHRelation() function")
        plot = Plotter.Plotter()

        MInit_all = []
        MFinal_all = []
        MDiff_all = []
        Mask_all = []
        
        # Looping over all simulations
        for i in range(4):
            # Loading in the sink particle data from binary files.
            self.BinaryReader = DataReader.BinaryReader(f"{self.base}{self.dir[i]}", feedback=self.feedback[i], array_length=350)
            SinkData = SinkParticles.SinkData(f"{self.base}{self.dir[i]}")
            
            #Filtering out sink particles to just PopIII particles.
            SinkData.filter_by_type("PopIII", kind="self")
            MInit = SinkData.M_init * 1e10 / CS.hubble_parameter
            MFinal = SinkData.M_final * 1e10 / CS.hubble_parameter
            MDiff = SinkData.M_diff * 1e10 / CS.hubble_parameter

            mask1 = MDiff > 0.1
            mask2 = MDiff <= 0.1
            print(MInit[mask1])

            # Plotting
            plot.ax.scatter(MInit[mask1], MFinal[mask1], color="black", marker="o", s=20, zorder=2)
            plot.ax.scatter(MInit[mask2], MFinal[mask2], color="red", marker="o", s=20, zorder=2, alpha=0.3)

            MInit_all.append(MInit)
            MFinal_all.append(MFinal)
            MDiff_all.append(MDiff)
            Mask_all.append(mask1.astype(int))  # 1 for black, 0 for red

        x = np.linspace(0, 400, 100)
        plot.ax.plot(x, x, alpha=0.2, zorder=0, color="black") 
        
        #Configuring the plots
        xmin, xmax = plt.xlim()
        plot.ax.axvspan(xmin, 40, color="gold", alpha=0.5, zorder=0)
        plot.ax.axvspan(142, 260, color="indigo", alpha=0.5, zorder=0)
        plot.ax.text(0.18, 0.4, "Type-II Supernova", rotation="vertical", transform=plot.ax.transAxes, fontsize=12)
        plot.ax.text(0.7, 0.02, "Direct Collapse BH", rotation="vertical", transform=plot.ax.transAxes, fontsize=12)
        plot.ax.text(0.82, 0.5, "Pair-Instability Supernova", rotation="vertical", transform=plot.ax.transAxes, fontsize=12)
        plot.ax.text(0.95, 0.02, "Direct Collapse BH", rotation="vertical", transform=plot.ax.transAxes, fontsize=12)

        plot.add_details(plot.ax, xscale="log", yscale="log", xlabel="M$_\mathrm{PopIII} [\mathrm{M}_\odot$]", ylabel="M$_\mathrm{BH,f} [\mathrm{M}_\odot]$", grid=False, xlim=(11, 350), ylim=(4, 50000))

        plot.add_legends(type="scatter", color="black", label="$\Delta \mathrm{M} > 0$")
        plot.add_legends(type="scatter", color="red", label="$\Delta \mathrm{M}$ <= 0")

        plot.save(f"{self.output_dir}Mehta_ED_Fig3.jpg", legend_loc="upper left")

        # ---------------------
        #  SOURCE DATA EXPORT
        # ---------------------

        MInit_all = np.concatenate(MInit_all)
        MFinal_all = np.concatenate(MFinal_all)
        MDiff_all = np.concatenate(MDiff_all)
        Mask_all = np.concatenate(Mask_all)

        # combine into one array
        data = np.column_stack([MInit_all, MFinal_all, MDiff_all, Mask_all])

        header = "MInit MFinal MDiff Mask(1=DeltaM>0.1)"

        np.savetxt(f"{self.output_dir}Mehta_SourceData_ED_Fig3.txt",
                data,
                header=header,
                comments='')

        print("Mehta_SourceData_ED_Fig3.txt written successfully.")

    def RadiationPressure(self):
        """
        Generates a plot for the final masses of BHs as a function of the ratio of radiation pressure flux and 
        inward falling gas flux. Also creates the radiation pressure binary file if not present.

        Produces:
        1. A scatter plot. (Mehta_ED_Fig8.jpg)
        2. Source data txt. (Mehta_SourceData_ED_Fig8.txt)
        """
        print("Entering RadiationPressure function.")

        def plots():
            # Initializing the plot
            plot = Plotter.Plotter()

            # Loading the data from a binary file
            RadiationData = self.PickleReader.pickle_reader(f"Level15/RadiationPressure.pkl")

            InwardGasFlux = np.abs(np.array([RadiationData[ID]["InwardGasFlux"] for ID in RadiationData]))
            RadiationFlux = np.array([RadiationData[ID]["RadiationFlux2"] for ID in RadiationData])
            MDiff = np.array([RadiationData[ID]["MGrowth"] for ID in RadiationData])

            ratio = InwardGasFlux / RadiationFlux
            
            plot.ax.scatter(ratio, MDiff, s=25, color="black", zorder=1)
            
            xmin, xmax = plot.ax.get_xlim()
            plot.ax.axvspan(xmin, 1, color="gold", alpha=0.5, zorder=0)
            plot.ax.axvspan(1, xmax, color="indigo", alpha=0.5, zorder=0)
            
            plot.add_details(yscale="log", xlabel="Gas Flux/Radiation Flux", ylabel="$\Delta \mathrm{M}_{Growth}$", xscale="log", xlim=(xmin, xmax))
            plot.save(f"{self.output_dir}Mehta_ED_Fig8.jpg")

            # ---------------------
            #  SOURCE DATA EXPORT
            # ---------------------

            data = np.column_stack([ratio, MDiff])

            header = "GasFlux_over_RadiationFlux   MDiff"

            np.savetxt(
                f"{self.output_dir}Mehta_SourceData_ED_Fig8.txt",
                data,
                header=header,
                comments=''
            )

            print("Mehta_SourceData_ED_Fig8.txt written successfully.")

            plot = Plotter.Plotter()

            FEdd = np.array([RadiationData[ID]["FEdd"] for ID in RadiationData])
            PhotonTrappingLuminosityFrac = 1 + np.log(FEdd)
            plot.ax.scatter(FEdd, PhotonTrappingLuminosityFrac, s=15, color="black")

            x = np.linspace(90, 300)
            
            y = 1 / 12 * x
            plot.ax.plot(x, y, color="grey")
            y = DU.PhotonTrappingLuminosity(x, mode=2)
            plot.ax.plot(x, y, color="red")
            y = DU.PhotonTrappingLuminosity(x, mode=3)
            plot.ax.plot(x, y, color="orange")
            y = DU.PhotonTrappingLuminosity(x, mode=4)
            plot.ax.plot(x, y, color="indigo")
            

            plot.add_details(xscale="log", yscale="log", xlabel="$\dot{M}/\dot{M}_{Edd}$", ylabel="L/L$_{\mathrm{Edd}}$")
            plot.add_legends(color="grey", label="Constant $\eta$")
            plot.add_legends(color="red", label="Ohsuga et al 2002. Model A")
            plot.add_legends(color="orange", label="Ohsuga et al 2002. Model B")
            plot.add_legends(color="indigo", label="Madau et al 2014")
            plot.add_legends(type="scatter", color="black", label="Our values.")
            plot.save(f"{self.output_dir}Luminosity.png")
        
            '''
            plot = Plotter.Plotter()
            plot.ax.scatter(CosmicTime[-1] - CosmicTime[-2], StellarMass[-1], color="black")
            plot.ax.scatter(0, StellarMass[-2], color="black")
            t = CosmicTime[-1] - CosmicTime[-2] # in yr
            print(f"Time duration is {t} yr")
            m0 = StellarMass[-2]
            
            FEdd = np.linspace(20, f_edd+20, 10)
            for j in range(len(FEdd)):
                time = np.linspace(0, t, N)
                mass = DU.SuperEddingtonGrowth(m0, time, f_edd=FEdd[j])
                plot.ax.plot(time, mass, color="grey", alpha=0.5, linestyle="--")
            plot.add_details(plot.ax, xlabel="Cosmic Time [yr]", ylabel="Sink Mass [M$_\odot$]", yscale="log", xlim=(0, (CosmicTime[-1] - CosmicTime[-2])*1.1), xscale="log")
            plot.save(f"{self.output_dir}RadiationPressureGrowth.png")
            '''

        files = os.listdir(f"/home/daxal/data/ProductionRuns/Rarepeak_zoom/Level15/")
        if "RadiationPressure.pkl" in files:
            plots()
            return

        SinkData = SinkParticles.SinkData(self.base)
        MDiff = SinkData.M_diff * 1e10 / CS.hubble_parameter
        mask = MDiff > 1.0e3
        MDiff = MDiff[mask]
        IDs = SinkData.ids[mask]
        RadiationData = {}
        
        
        for ID in IDs:
            RadiationData[ID] = {}
            SinkDataID = SinkData.sink_particles[ID]    
            evolution = SinkDataID["evolution"]
            times = np.array(list(sorted(evolution.keys())))
            Redshift = 1 / times - 1
            CosmicTime = cosmo.age(Redshift).to("yr").value
            FormationSnap = SinkDataID["meta"]["FormationSnap"]
            StellarMass = np.array([evolution[t]["StellarMass"] for t in times]) * 1e10 / CS.hubble_parameter
            SinkPos = np.array([evolution[t]["Pos"] for t in times])
            SinkVel = np.array([evolution[t]["Vel"] for t in times])
            MassDiff = np.diff(StellarMass)
            ind = np.argsort(MassDiff)
            GrowthSnapIndex = ind[-1]
            GrowthSnap = FormationSnap + GrowthSnapIndex

            print(f"Max Mass Diff: {MassDiff[GrowthSnapIndex]} at time {times[GrowthSnapIndex]}, z = {1/times[GrowthSnapIndex] - 1} at snap {GrowthSnap}")
            
            M1, M2 = StellarMass[GrowthSnapIndex], StellarMass[GrowthSnapIndex + 1]
            T1, T2 = CosmicTime[GrowthSnapIndex], CosmicTime[GrowthSnapIndex + 1]
            f_edd = DU.FindEddingtonFactor(M1, M2, T2 - T1)
            RadiationData[ID]["MGrowth"] = M2 - M1
            RadiationData[ID]["GrowthTime"] = T2 - T1
            RadiationData[ID]["FEdd"] = f_edd
            print(f"Required Eddington factor for such growth is {f_edd}")
            
            MeanMass = np.sqrt(M1 * M2)
            RadiationForce = DU.RadiationPressure(MeanMass, f_edd, mode=1, PhotonTrapping=True)
            RadiationMomentum = RadiationForce * (T2 - T1) * CS.yr_to_s

            RadiationData[ID]["RadiationFlux1"] = RadiationForce
            RadiationData[ID]["RadiationMomentum1"] = RadiationMomentum

            print(f"Radiation Force: {RadiationForce} g cm/s^2, Radiation Momentum: {RadiationMomentum} g cm/s. Without gas coupling.")

            sfile = f"{self.base}snapdir_{GrowthSnap:03d}/snap_{GrowthSnap:03d}.0.hdf5"
            ds = yt.load(sfile)
            ad = ds.all_data()
            
            SinkPos = SinkPos[GrowthSnapIndex]
            SinkVel = SinkVel[GrowthSnapIndex] * pow(times[GrowthSnapIndex], 0.5)
            
            YTU = YT.YTUtils(ds, ad)
            Radius = 0.5
            YTU.SelectRegion(SinkPos, Radius, units="pc")
            print("Region found.")
            #GasFields = [("PartType0", "Coordinates"), ("PartType0", "Masses"), ("PartType0", "Velocities")]
            #GasData = YTU.region.to_astropy_table(GasFields)
            GasPos = YTU.region[("PartType0", "Coordinates")].value
            
            GasMass = YTU.region[("PartType0", "Masses")].in_units("Msun").value
            GasVelocity = YTU.region[("PartType0", "Velocities")].in_units("km/s").value

            TotalGasMass = np.sum(GasMass)
            
            Sigma = TotalGasMass * CS.Msun_to_g / (np.pi * (Radius * CS.pc_in_cgs)**2) # in g/cm^2
            KappaES = 0.34
            TauES = KappaES * Sigma
            Frac = 1.0 - np.exp(-TauES)
            
            RadiationForce *= Frac
            RadiationMomentum *= Frac
            print(f"Radiation Force: {RadiationForce} g cm/s^2, Radiation Momentum: {RadiationMomentum} g cm/s. With gas coupling.")
            
            RadiationData[ID]["RadiationFlux2"] = RadiationForce
            RadiationData[ID]["RadiationMomentum2"] = RadiationMomentum
            
            dr = (GasPos - SinkPos) * 1e3 / CS.hubble_parameter / (1 + ds.current_redshift) # in pc
            r = np.linalg.norm(dr, axis=1) # in pc
            dv = GasVelocity - SinkVel
            vrad = np.sum(dr * dv, axis=1) / r # in km/s
            mask = vrad < 0
            print(f"Number of gas cells in the region is {len(GasPos)}. Total Mass of the gas is: {TotalGasMass} Msun.")
            GasMass = GasMass[mask]
            InwardGasMass = np.sum(GasMass)
            print(f"Number of gas cells in the region is {len(GasMass)}. Total Mass of the gas is: {InwardGasMass} Msun.")
            RadiationData[ID]["TotalGasMass"] = TotalGasMass
            RadiationData[ID]["InwardGasMass"] = InwardGasMass
            vrad = vrad[mask]
            if M1 < TotalGasMass:
                tff = DU.FreeFallTimescale(TotalGasMass, Radius, mode=1)
            else:
                tff = DU.FreeFallTimescale(TotalGasMass, Radius, mode=2)
            RadiationData[ID]["FreeFallTime"] = tff   
            ForceGasInfall = np.sum(GasMass * vrad) * CS.Msun_to_g * 1e5 / tff
            Momentum = np.sum(GasMass * vrad) * CS.Msun_to_g * 1e5 # in g cm/s
            RadiationData[ID]["InwardGasFlux"] = ForceGasInfall
            RadiationData[ID]["InwardGasMomentum"] = Momentum
            print(f"Inward falling gas force: {ForceGasInfall} g cm/s^2 and momentum: {Momentum} g cm/s")
        
        with open(f"{self.base}RadiationPressure.pkl", "wb") as f:
            pickle.dump(RadiationData, f)
        
        plots()
  
    def NumberDensityEvolution(self):
        """
        Generates the evolution of the number density of heavy seeds along with redshift.

        Produces:
        1. A line plot. (Mehta_ED_Fig9.jpg)
        2. Source data txt. (Mehta_SourceData_ED_Fig9.txt)
        """
        print("Entering the NumberDensityEvolution() function.")
        plot = Plotter.Plotter()
        Volume = [1**3, 1**3, 0.5**3, 0.5**3]

        colors = color_schemes["Material"]

        AllRedshift = []
        AllHeavy = []
        AllGalaxy = []

        #Looping over all four simulations
        for i in range(4):

            # Loading in the reduced sink particle data from binary files.
            SinkData = SinkParticles.SinkData(self.base + self.dir[i])
            MDiff = SinkData.M_diff * 1e10 / CS.hubble_parameter
            MInit = SinkData.M_init * 1e10 / CS.hubble_parameter
            MFinal = SinkData.M_final * 1e10 / CS.hubble_parameter

            ind = np.where((MFinal > 1e3) & (MDiff > MInit))[0]
            ids = SinkData.ids[ind]
            print(ids)
            print(f"Number of heavy seeds in {self.dir[i]} is {len(ids)}")
            start_snap, end_snap = DU.file_range(f"{self.base}{self.dir[i]}")
            HeavySeedNumber = np.zeros(end_snap - start_snap, dtype=float)
            GalaxyNumber = np.zeros(end_snap - start_snap, dtype=float)
            Redshift = np.zeros(end_snap - start_snap, dtype=float)
            for snap in range(start_snap, end_snap):
                if i == 0 and snap == 32: continue
                self.BinaryReader = DataReader.BinaryReader(f"{self.base}{self.dir[i]}", feedback=self.feedback[i], array_length=350)
                self.BinaryReader.read_sink_snap(snap)
                if self.BinaryReader.time == 0: continue
                
                Redshift[snap - start_snap] = 1 / self.BinaryReader.time - 1
                if self.BinaryReader.num_sinks == 0: continue
                Groups = GroupsReader.Reader(f"{self.base}{self.dir[i]}", snap)
                Halos = Groups.halos
                HaloMass = Halos["GroupMass"] * 1e10 / CS.hubble_parameter
                SinkMass = Halos["GroupMassType"][:, 5] * 1e10 / CS.hubble_parameter

                mask = SinkMass > 0
                GalaxyNumber[snap - start_snap] = len(HaloMass[mask])
                
                Type = self.BinaryReader.extract_data("Type")
                ID5 = self.BinaryReader.extract_data("ID")
                Mass = self.BinaryReader.extract_data("StellarMass") * 1e10 / CS.hubble_parameter
                for k, ID in enumerate(ids):
                    if ID not in ID5: continue
                    mask = ID5 == ID
                    if Type[mask] == 3 :
                        HeavySeedNumber[snap - start_snap] += 1          

            #norm = Groups.Normalization(Redshift[-1])

            #HeavySeedNumber[:] *= norm
            #GalaxyNumber[:] *= norm
            
            mask = HeavySeedNumber != 0
            plot.ax.plot(Redshift[mask], HeavySeedNumber[mask] / Volume[i], color=self.colors[i])
            mask = GalaxyNumber != 0
            plot.ax.plot(Redshift[mask], GalaxyNumber[mask] / Volume[i], color=self.colors[i], linestyle="--")

            AllRedshift.append(Redshift)
            AllHeavy.append(HeavySeedNumber / Volume[i])
            AllGalaxy.append(GalaxyNumber / Volume[i])
        
        # Configuring the plot.
        plot.ax.invert_xaxis()
        plot.add_details(plot.ax, xlabel="Redshift", ylabel="dN/dV [cMpc$^{-3}$]", yscale="log", xlim=(34, 14))
        plot.add_legends(color=self.colors[0], label="L13")
        plot.add_legends(color=self.colors[1], label="L14")
        plot.add_legends(color=self.colors[2], label="L15")
        plot.add_legends(color=self.colors[3], label="L15_BHFB")
        plot.add_legends(color="black", label="MBH")
        plot.add_legends(color="black", label="Galaxy", linestyle="--")

        plot.save(f"{self.output_dir}Mehta_ED_Fig9.jpg", legend_loc="center right")  

        # ---------------------
        #  SOURCE DATA EXPORT
        # ---------------------  

        print("Writing source data for ED Figure 9...")

        # Build one combined table
        # Column format:
        # Redshift_0 Heavy_0 Galaxy_0  Redshift_1 Heavy_1 Galaxy_1  ... (all 4 sims)

        max_len = max(len(r) for r in AllRedshift)

        def pad(arr):
            """Pad arrays with NaN to match the longest length."""
            out = np.full(max_len, np.nan)
            out[:len(arr)] = arr
            return out

        combined = np.column_stack([
            pad(AllRedshift[0]), pad(AllHeavy[0]), pad(AllGalaxy[0]),
            pad(AllRedshift[1]), pad(AllHeavy[1]), pad(AllGalaxy[1]),
            pad(AllRedshift[2]), pad(AllHeavy[2]), pad(AllGalaxy[2]),
            pad(AllRedshift[3]), pad(AllHeavy[3]), pad(AllGalaxy[3]),
        ])

        header = (
            "Redshift_L13   HeavySeedDensity_L13   GalaxyDensity_L13   "
            "Redshift_L14   HeavySeedDensity_L14   GalaxyDensity_L14   "
            "Redshift_L15   HeavySeedDensity_L15   GalaxyDensity_L15   "
            "Redshift_L15BHFB   HeavySeedDensity_L15BHFB   GalaxyDensity_L15BHFB"
        )

        np.savetxt(
            f"{self.output_dir}Mehta_SourceData_ED_Fig9.txt",
            combined,
            header=header,
            comments=''
        )

        print("Mehta_SourceData_ED_Fig9.txt written successfully.")

    def LargeProjectionPlots(self, mode=1):
        """
        Generates a large composite plot of gas density projection or gas temperature projections.
        mode = 1: Gas density projection.
        mode = 2: Gas temperature projection.

        Produces:
        1. A multi-panel composite plot (Mehta_Fig2.pdf or Mehta_ED_Fig4.jpg)
        2. Source data txt. (Mehta_SourceData_Fig2.txt or Mehta_SourceData_ED_Fig4.txt)
        """
        print("Entering the LargeProjectionPlots() function")

        # ID of the largest growing BH in L15_BHFB simulation.
        ID = 393972273.0
        self.base = "/home/daxal/data/ProductionRuns/Rarepeak_zoom/Level15_feedback/"
        self.colors = color_schemes["Material"]
        files = sorted(glob.glob(f"{self.base}gas_fields_*.hdf5"))
        if not files:
            print(f"No files found for particle ID {ID} in directory {self.base}")
            return
        
        if mode == 1:
            fname = f"{self.output_dir}Mehta_SourceData_Fig2.txt"
        else:
            fname = f"{self.output_dir}Mehta_SourceData_ED_Fig4.txt"
        ff = open(fname, "w")
        ff.write("------------------------------------------------------------\n")
        if mode == 1:
            ff.write("Panel A - Density Projection\n")
        else:
            ff.write("Panel A - Temperature Projection\n")
        ff.write("------------------------------------------------------------\n")
        ff.write("# Columns: x_pix\ty_pix\tdensity\n")
        
        self.BinaryReader = DataReader.BinaryReader(f"{self.base}", feedback=True, array_length=350)
        Fig = plt.figure(figsize=(10, 10))
        snaps = [63, 66, 69, 70]

        DensityMax = 1e-17
        DensityMin = 1e-26
        TemperatureMin = 1e2
        TemperatureMax = 1e5
        
        MainFig = Fig.add_axes([0.0, 0.0, 0.6, 0.6]) #[left, bottom, width, height]
        MainFig.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        if mode == 1:
            Cmap = mpl.cm.get_cmap("Greys").copy()
        else:
            Cmap = mpl.cm.get_cmap("Reds").copy()
        Cmap.set_under(alpha=0.0)
        
        # Reading in the gas fields from a reduced dataset.
        with h5py.File(f"{self.base}gas_fields_{ID}_2500_063.hdf5", "r") as f:
            frb_density = f["Density"][:]
            frb_temperature = f["Temperature"][:]
            redshift = f.attrs["redshift"]

            if mode == 1:
                MainFig.imshow(frb_density, cmap=Cmap, norm=mpl.colors.LogNorm(vmin=DensityMin, vmax=DensityMax), origin="lower")
                Ny, Nx = frb_density.shape
                for iy in range(Ny):
                    for ix in range(Nx):
                        ff.write(f"{ix}\t{iy}\t{frb_density[iy, ix]:.6e}\n")
            else:
                MainFig.imshow(frb_temperature, cmap=Cmap, norm=mpl.colors.LogNorm(vmin=TemperatureMin, vmax=TemperatureMax), origin="lower")
                Ny, Nx = frb_temperature.shape
                for iy in range(Ny):
                    for ix in range(Nx):
                        ff.write(f"{ix}\t{iy}\t{frb_temperature[iy, ix]:.6e}\n")

            MainFig.set(xlim=(0, 800), ylim=(0, 800))
        
        ScaleLine = plt.Line2D((0.5, 0.5), (0.5, 0.5), color="k", linewidth=2)
        ScaleLine.set_xdata([0.52, 0.58])
        ScaleLine.set_ydata([0.02, 0.02])

        Fig.add_artist(ScaleLine)

        MainFig.text(0.875, 0.05, "500pc", transform=MainFig.transAxes, color="black")

        MainFig.text(0.05, 0.95, f"a", transform=MainFig.transAxes, fontsize=15, fontweight="bold", va="top", ha="right", color="k")

        Line1 = plt.Line2D((0.5, 0.5), (0.5, 0.5), color="k", linewidth=1)
        Line1.set_xdata([0.27, 0.00])
        Line1.set_ydata([0.33, 0.60])
        Fig.add_artist(Line1)

        Line2 = plt.Line2D((0.5, 0.5), (0.5, 0.5), color="k", linewidth=1)
        Line2.set_xdata([0.33, 0.40])
        Line2.set_ydata([0.33, 0.60])
        Fig.add_artist(Line2)

        Rect = mpl.patches.Rectangle((0.27, 0.27), 0.06, 0.06, fill=False, edgecolor="k", linewidth=1)
        Fig.add_artist(Rect)

        ff.write("------------------------------------------------------------\n")
        if mode == 1:
            ff.write("Panel B - Density Projection\n")
        else:
            ff.write("Panel B - Temperature Projection\n")
        ff.write("------------------------------------------------------------\n")
        ff.write("# Columns: x_pix\ty_pix\tdensity\n")

        Zoom1 = Fig.add_axes([0.0, 0.6, 0.4, 0.4])
        Zoom1.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        with h5py.File(f"{self.base}gas_fields_{ID}_250_063.hdf5", "r") as f:
            frb_density = f["Density"][:]
            frb_temperature = f["Temperature"][:]
            redshift = f.attrs["redshift"]
            if mode == 1:
                Zoom1.imshow(frb_density, cmap=Cmap, norm=mpl.colors.LogNorm(vmin=DensityMin, vmax=DensityMax), origin="lower")
                Ny, Nx = frb_density.shape
                for iy in range(Ny):
                    for ix in range(Nx):
                        ff.write(f"{ix}\t{iy}\t{frb_density[iy, ix]:.6e}\n")
            else:
                Zoom1.imshow(frb_temperature, cmap=Cmap, norm=mpl.colors.LogNorm(vmin=TemperatureMin, vmax=TemperatureMax), origin="lower")
                Ny, Nx = frb_temperature.shape
                for iy in range(Ny):
                    for ix in range(Nx):
                        ff.write(f"{ix}\t{iy}\t{frb_temperature[iy, ix]:.6e}\n")

            Zoom1.set(xlim=(0, 800), ylim=(0, 800))

        ScaleLine = plt.Line2D((0.5, 0.5), (0.5, 0.5), color="k", linewidth=2)
        ScaleLine.set_xdata([0.33, 0.37])
        ScaleLine.set_ydata([0.62, 0.62])

        Fig.add_artist(ScaleLine)
        Zoom1.text(0.83, 0.07, "50pc", transform=Zoom1.transAxes, color="black")

        Zoom1.text(0.07, 0.95, f"b", transform=Zoom1.transAxes, fontsize=15, fontweight="bold", va="top", ha="right", color="k")

        Line1 = plt.Line2D((0.5, 0.5), (0.5, 0.5), color="k", linewidth=1)
        Line1.set_xdata([0.22, 0.60])
        Line1.set_ydata([0.82, 1.00])
        Fig.add_artist(Line1)

        Line2 = plt.Line2D((0.5, 0.5), (0.5, 0.5), color="k", linewidth=1)
        Line2.set_xdata([0.22, 0.60])
        Line2.set_ydata([0.78, 0.75])
        Fig.add_artist(Line2)

        Rect = mpl.patches.Rectangle((0.18, 0.78), 0.04, 0.04, fill=False, edgecolor="k", linewidth=1)

        Fig.add_artist(Rect)

        Zoom2 = Fig.add_axes([0.6, 0.75, 0.25, 0.25])
        Zoom3 = Fig.add_axes([0.6, 0.5, 0.25, 0.25])
        Zoom4 = Fig.add_axes([0.6, 0.25, 0.25, 0.25])
        Zoom5 = Fig.add_axes([0.6, 0.0, 0.25, 0.25])

        axZoom = [Zoom2, Zoom3, Zoom4, Zoom5]
        text = ["c", "d", "e", "f"]
        Panel = ["C", "D", "E", "F"]

        for i, ax in enumerate(axZoom):
            with h5py.File(f"{self.base}gas_fields_{ID}_{snaps[i]:03d}.hdf5", "r") as f:
                ff.write("------------------------------------------------------------\n")
                if mode == 1:
                    ff.write(f"Panel {Panel[i]} - Density Projection\n")
                else:
                    ff.write(f"Panel {Panel[i]} - Temperature Projection\n")
                ff.write("------------------------------------------------------------\n")
                ff.write("# Columns: x_pix\ty_pix\tdensity\n")
                redshift = f.attrs["redshift"]

                # Reading the binary file for sink particles for the required snapshots.
                self.BinaryReader.read_sink_snap(snaps[i])
                pos5 = self.BinaryReader.extract_data("Pos") * 1e3 / (1 + redshift) / CS.hubble_parameter
                Type = self.BinaryReader.extract_data("Type")
                mass5 = self.BinaryReader.extract_data("StellarMass") * 1e10 / CS.hubble_parameter
                center = f.attrs["center"] * 1e3 / (1 + redshift) / CS.hubble_parameter #in pc

                dist = np.linalg.norm(pos5 - center, axis=1)

                radius = f.attrs["width"]
                mask = dist < radius
                pos5 = pos5[mask]
                Type = Type[mask]
                mask = dist == 0.0
                mass5 = mass5[mask].item()

                normal = f.attrs["normal"]
                if normal == "x":
                    pos5 = pos5[:, [1, 2]]
                    center = center[[1, 2]]
                elif normal == "y":
                    pos5 = pos5[:, [0, 2]]
                    center = center[[0, 2]]
                elif normal == "z":
                    pos5 = pos5[:, [0, 1]]
                    center = center[[0, 1]]
                
                frb_density = f["Density"][:]
                frb_temperature = f["Temperature"][:]

                buff_size = frb_density.shape[0]
                radius_in_pixles = buff_size / 2

                p5_x = (pos5[:, 0] - center[0]) / radius * radius_in_pixles + radius_in_pixles
                p5_y = (pos5[:, 1] - center[1]) / radius * radius_in_pixles + radius_in_pixles

                frb_density[frb_density < DensityMin] = DensityMin

                if mode == 1:
                    ax.imshow(frb_density, cmap=Cmap, norm = mpl.colors.LogNorm(vmin=DensityMin, vmax=DensityMax), origin="lower")
                    Ny, Nx = frb_density.shape
                    for iy in range(Ny):
                        for ix in range(Nx):
                            ff.write(f"{ix}\t{iy}\t{frb_density[iy, ix]:.6e}\n")
                else: 
                    ax.imshow(frb_temperature, cmap=Cmap, norm=mpl.colors.LogNorm(vmin=TemperatureMin, vmax=TemperatureMax), origin="lower", alpha=1.0)
                    Ny, Nx = frb_temperature.shape
                    for iy in range(Ny):
                        for ix in range(Nx):
                            ff.write(f"{ix}\t{iy}\t{frb_temperature[iy, ix]:.6e}\n")
                # Over plotting the sink particles.
                mask = Type == 3
                ax.scatter(p5_x[mask], p5_y[mask], color=self.colors[0], marker="x", s=50)
                mask = Type == 0
                ax.scatter(p5_x[mask], p5_y[mask], color=self.colors[1], marker="+", s=50)

                ff.write("# Overlaying Sink Particle Data\n")
                ff.write("# Columns: x\ty\tType\n")
                for x, y, t in zip(p5_x, p5_y, Type):
                    ff.write(f"{x}\t{y}\t{t}\n")
                ax.set(xlim=(0, 800), ylim=(0, 800))


                ax.text(0.55, 0.9, f"z = {redshift:.2f}", transform=ax.transAxes, color="black")
                ax.text(0.05, 0.05, "M$_\mathrm{BH}$ = %.2f M$_\odot$" %mass5, transform=ax.transAxes, color="black")
                ax.text(0.05, 0.95, f"{text[i]}", transform=ax.transAxes, fontsize=15, fontweight="bold", va="top", ha="right", color="k")

                if i == 0:
                    ScaleLine = plt.Line2D((0.5, 0.5), (0.5, 0.5), color="k", linewidth=2)
                    ScaleLine.set_xdata([0.8, 0.825])
                    ScaleLine.set_ydata([0.77, 0.77])
                    Fig.add_artist(ScaleLine)
                    ax.text(0.80, 0.1, "5pc", transform=ax.transAxes, color="black")
                
                ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        if mode == 1:
            DensityNorm = mpl.colors.LogNorm(vmin=DensityMin, vmax=DensityMax)
            ColorBar1 = Fig.add_axes([0.88, 0.025, 0.025, 0.95])
            cbar1 = mpl.colorbar.ColorbarBase(ColorBar1, cmap="Greys", norm=DensityNorm)
            cbar1.set_label('Density (g/cm³)', fontsize=15)
        else:
            TemperatureNorm = mpl.colors.LogNorm(vmin=TemperatureMin, vmax=TemperatureMax)
            ColorBar2 = Fig.add_axes([0.88, 0.025, 0.025, 0.95])
            cbar2 = mpl.colorbar.ColorbarBase(ColorBar2, cmap="Reds", norm=TemperatureNorm, alpha=1.0)
            cbar2.set_label('Temperature (K)', fontsize=15)

        legend_elements = [
            Line2D([0], [0], marker="x", color=self.colors[0], label="Black Holes", markerfacecolor=self.colors[0], markersize=10, linestyle="none"),
            Line2D([0], [0], marker="+", color=self.colors[1], label="PopIII stars", markerfacecolor=self.colors[1], markersize=10, linestyle="None"),
        ]
        Fig.legend(handles=legend_elements, loc="center", bbox_to_anchor=(0.50, 0.82), frameon=False, fontsize=12)

        if mode == 1:
            Fig.savefig(f"{self.output_dir}Mehta_Fig2.pdf", dpi=300)
        else:
            Fig.savefig(f"{self.output_dir}Mehta_ED_Fig4.jpg", dpi=300)
        
        ff.close()
            
                                   
###############################

# "base" is the location where you downloaded all the data from figshare. Change it accordingly
base = "/home/daxal/data/ProductionRuns/Rarepeak_zoom/"
Script = Scripts(base)            
#Script.SinkHistogram()              ## Extended Data Fig 1
#Script.SinkGrowth()                 ## Main Text Fig 1, 3 and Extended Data Fig 5
#Script.HMF()                        ## Extended Data Fig 2
#Script.PopIIIBHRelation()           ## Extended Data Fig 3
#Script.SurroundingGasProperties()   ## Main Text Fig4 and Extended Data Fig 6
#Script.HostHaloProperties()         ## Extended Data Fig 7
#Script.NumberDensityEvolution()     ## Extended Data Fig 9
#Script.RadiationPressure()          ## Extended Data Fig 8
#Script.LargeProjectionPlots(mode=2) ##mode=1 for density, 2 for temperature## Main Text Fig2 and Extended Data Fig 4
