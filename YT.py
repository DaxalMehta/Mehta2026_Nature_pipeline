import yt
import numpy as np
import Utilities

DU = Utilities.DataUtilities()

class YTUtils:
    def __init__(self, ds, ad):
        self.ds = ds
        self.ad = ad
    
    def SelectRegion(self, center, radius, units="kpc"):
        radius = self.ds.quan(radius, units)
        center = self.ds.arr(center, "code_length")
        self.region = self.ds.sphere(center, radius)

        return self.region
    
    def _Metallicity(self, field, data):
        Metallicity = np.divide(data[("PartType0", "SneTracerField")], data[("PartType0", "Masses")].v)
        Metallicity = Metallicity
        Metallicity[Metallicity < 1e-10] = 1e-10
        Metallicity = np.nan_to_num(Metallicity, nan=1e-10, posinf=1e-10, neginf=1e-10)
        
        return Metallicity
    
    def DMfields(self):
        self.ds.add_deposited_particle_field(("PartType1", "Masses"), method="cic", weight_field=None)

    def FindBHandStellarProperty(self, id_binary, type_binary):
        id5 = self.region[("PartType5", "ParticleIDs")].value
        mass5 = self.region[("PartType5", "Masses")].in_units("Msun").value
        pos5 = self.region[("PartType5", "Coordinates")]
        dist = np.linalg.norm((pos5 - self.region.center), axis=1)

        sorter = np.argsort(id_binary)
        idx = np.searchsorted(id_binary, id5, sorter=sorter)
        mask = (idx >= 0) & (idx < len(type_binary))
        Type = np.zeros(len(id5))
        Type[mask] = type_binary[sorter[idx[mask]]]

        is_bh = Type == 3
        is_stellar = np.isin(Type, [0, 2])

        BHMass = np.sum(mass5[is_bh])
        StellarMass = np.sum(mass5[is_stellar])

        if "PartType4" in self.ds.fields:
            StellarMass += self.region[("PartType4", "Masses")].in_units("Msun").sum().to_value()
        
        massive_mask = is_bh & (mass5 > 1e5)
        ID = [int(DU.unwrap(pid)) for pid in id5[massive_mask]]
        HeavySeedMass = [DU.unwrap(pmass) for pmass in mass5[massive_mask]]
        Separation = [DU.unwrap(pdist) for pdist in dist[massive_mask]]
        
        if np.any(massive_mask):
            i_max = np.argmax(mass5[is_bh])
            MostMassiveSeed = mass5[is_bh][i_max]
            MostMassiveID = id5[is_bh][i_max]
        else:
            MostMassiveSeed, MostMassiveID = 0.0, -1
        
        return BHMass, StellarMass, ID, HeavySeedMass, Separation, MostMassiveSeed, int(MostMassiveID)
    
    def CheckContamination(self):
        if "PartTyp2" in self.ds.fields:
            Mass2 = self.region[("PartType2", "Masses")].in_units("Msun").value
            if np.sum(Mass2) > 0:
                print("Warning: Contamination particles found in the region. Contanimation Mass = %.2e Msun" % np.sum(Mass2))



class YTPlotter(YTUtils):
    def __init__(self, ds, ad, normal="x"):
        super().__init__(ds, ad)
        self.normal = normal
        self.plots = []
        self.fields = []
        self.region = None
        self.center = ds.domain_center
        self.width = "1000"
    
    def add_field(self, field, weight_field=("gas", "number_density")):
        self.fields.append((field, weight_field))
    
    def add_metallicity(self, weight_field=("PartType0", "Masses")):
        self.ds.add_field(("PartType0", "Metallicity"), units="", sampling_type="particle", function=self._Metallicity)
        field = ("PartType0", "Metallicity")
        self.fields.append((field, weight_field))
    
    def set_center(self, center=None, width=None, units="kpc"):
        if units == "code_length":
            width = width / (1 + self.ds.current_redshift) / self.ds.hubble_constant
            units = "kpc"
            if width > 1500:
                width /= 1000
                units = "Mpc"
        if center is not None:
            self.center = self.ds.arr(center, "code_length")
        self.width = self.ds.quan(width, units)
        self.region = self.ds.sphere(self.center, self.width)
    
    def plot_sink_particles(self, id_binary, type_binary, heavy_only=False, stars=False):
        id5 = self.region[("PartType5", "ParticleIDs")].value
        pos5 = self.region[("PartType5", "Coordinates")]
        mass5 = self.region[("PartType5", "Masses")].in_units("Msun").value
        print(len(pos5))
        Type = np.zeros(len(pos5))
        for i in range(len(id5)):
            ind = id_binary == id5[i]
            Type[i] = type_binary[ind]
        
        for i, result in enumerate(self.plots):
            for field, plot in result.items():
                for j in range(len(id5)):
                    color = {0: "cyan", 2: "gold", 3: "black"}.get(Type[j], None)
                    size = {0: 2, 2: 2, 3: 5}.get(Type[j], 1)
                    if (color == "black" and mass5[j] > 1e5): color = "red"
                    if heavy_only and color != "red": continue
                    if color:
                        plot.annotate_marker(pos5[j], coord_system="data", color=color, s=size, marker="o")
        if stars and "PartType4" in self.ds.fields:
            id4 = self.region[("PartType4", "ParticleIDs")].value
            Pos4 = self.region[("PartType4", "Coordinates")]
            for i, result in enumerate(self.plots):
                for field, plot in result.items():
                    for j in range(len(id4)):
                        print("Annotating PartType4.")
                        plot.annotate_marker(Pos4[j], coord_system="data", color="g", s=2, marker="o")

    def plot_halo(self, halo_center, halo_radius, units="kpc"):
        if units == "code_length":
            halo_radius = halo_radius / (1 + self.ds.current_redshift) / self.ds.hubble_constant
            units = "kpc"
        radius = self.ds.quan(halo_radius, units)
        center = self.ds.arr(halo_center, "code_length")
        for i, result in enumerate(self.plots):
            for field, plot in result.items():
                plot.annotate_sphere(center, radius=radius, circle_args={"color": "red"})
    
    def customize(self, cmap=None, background=False, lowlim=None, highlim=None, units=None, buff_size=None, beauty=False):
        for i, result in enumerate(self.plots):
            for field, plot in result.items():
                if buff_size: plot.set_buff_size(buff_size)
                if cmap: plot.set_cmap(field, cmap)
                if background == True: plot.set_background_color(field)
                if units: plot.set_unit(field, units)
                if lowlim: plot.set_zlim(field, lowlim, highlim)

                if beauty == True:
                    plot.hide_colorbar()
                    plot.hide_axes(draw_frame=True)           

    def annotate_text(self, locx, locy, text, fontsize=50, fontcolor="white"):
        for i, result in enumerate(self.plots):
            for field, plot in result.items():
                plot.annotate_text((locx, locy), text, coord_system="axis", text_args={"color":fontcolor, "fontsize":fontsize, "fontweight":"bold"})
    
    def save_plots(self, base="", prefix=""):
        import os
        os.makedirs(base, exist_ok=True)
        for i, result in enumerate(self.plots):
            for field, plot in result.items():
                fname = f"{prefix}_{i}_{field[1].replace(' ', '_')}.png"
                print(fname)
                plot.save(os.path.join(base, fname))
                print(f"Plot created at {base}{fname}")

class ProjectionPlot(YTPlotter):
    def plot(self):
        if len(self.fields) == 0: self.fields.append((("gas", "number_density"), ("gas", "number_density")))
        results = {}

        for field, weight in self.fields:
            p = yt.ProjectionPlot(self.ds, self.normal, field, center=self.center, width=self.width, weight_field=weight, data_source=self.region)
        
            results[field] = p
        self.plots.append(results)

        return results

class SlicePlot(YTPlotter):
    def plot(self):
        results = {}
        for field, weight in self.fields:
            s = yt.SlicePlot(self.ds, self.normal, field, center=self.center, width=self.width, weight_field=weight, data_source=self.region)

            results[field] = s
        self.plots.append(results)

        return results

class PhasePlot(YTPlotter):
    def plot(self):
        results = {}
        ph = yt.PhasePlot(self.region, ("gas", "number_density"), ("gas", "temperature"), ("gas", "mass"))
        results[("gas", "number_density")] = ph
        self.plots.append(results)
        return results
    
class ParticlePlot(YTPlotter):
    def plot(self, cmap="magma"):
        results = {}
        p = yt.ParticlePlot(self.ds, ("PartType1", "particle_position_y"), ("PartType1", "particle_position_z"), ("PartType1", "Masses"), center=self.center, width = self.width, data_source=self.region)

        p.set_unit(("PartType1", "Masses"), "Msun")
        p.set_cmap(("PartType1", "Masses"), cmap)
        p.set_background_color(("PartType1", "Masses"))

        results[("PartType1", "particle_position_x")] = p
        self.plots.append(results)

        return results

class ParticleProjectionPlot(YTPlotter):
    def plot(self, cmap="magma"):
        results = {}
        for field, weight in self.fields: 
            p = yt.ParticleProjectionPlot(self.ds, self.normal, field, weight_field=weight, deposition="cic", center=self.center, width=self.width)
    

        results[field] = p
        self.plots.append(results)

        return results

class StellarParticlePlot(YTPlotter):
    def plot(self, cmap="magma"):
        results = {}
        p = yt.ParticlePlot(self.ds, ("PartType4", "particle_position_y"), ("PartType4", "particle_position_z"), ("PartType4", "Masses"), center=self.center, width = self.width, data_source=self.region)

        p.set_unit(("PartType4", "Masses"), "Msun")
        p.set_cmap(("PartType4", "Masses"), cmap)
        p.set_background_color(("PartType4", "Masses"))

        results[("PartType4", "particle_position_x")] = p
        self.plots.append(results)

        return results

'''

    def change_cmap(self, cmap, background=True):
        for i, result in enumerate(self.plots):
            for field, plot in result.items():
                plot.set_cmap(field, cmap)
                if background:
                    plot.set_background_color(field)
    
    def set_zlim(self, lowlim, highlim):
        for i, result in enumerate(self.plots):
            for field, plot in result.items():
                plot.set_zlim(field, lowlim, highlim)
    
    def set_unit(self, units=""):
        for i , result in enumerate(self.plots):
            for field, plot in result.items():
                plot.set_unit(field, units)
    
    def buff_size(self, size):
        for i, result in enumerate(self.plots):
            for field, plot in result.items():
                plot.set_buff_size(size)
    
'''