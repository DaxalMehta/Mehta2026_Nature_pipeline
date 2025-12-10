import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import integrate


class Constants:
    def __init__(self):
        self.Msun_to_g = 1.989e33
        self.yr_to_s = 365 * 24 * 3600

        self.c_in_cgs = 3e10
        self.G_in_cgs = 6.67e-8
        self.k_B_in_cgs = 1.38e-16
        self.m_p_in_cgs = 1.67e-24
        self.pc_in_cgs = 3.086e18

        self.c_in_SI = 3e8
        self.G_in_SI = 6.67e-11
        self.k_B_in_SI = 1.38e-23
        self.m_p_in_SI = 1.67e-27
        self.pc_in_SI = 3.086e15

        self.hubble_parameter = 0.674

CS = Constants()
class DataUtilities:
    def __init__(self):
        pass

    def file_range(self, base):
        files = os.listdir(base)
        snapshot_files = [file for file in files if file.startswith("sink_snap_")]
        if not snapshot_files:
            snapshot_files = [file for file in files if file.startswith("groups_")]
            if not snapshot_files:
                return None, None
        snapshot_numbers = [int(file.split("_")[-1]) for file in snapshot_files]
        start_snap = min(snapshot_numbers)
        end_snap = max(snapshot_numbers) + 1
        return start_snap, end_snap
    
    def Type(self, t):
        if t == 0:
            return "PopIII"
        if t == 2:
            return "PopII"
        if t == 3:
            return "MBH"
    
    def SNeType(self, mass):
        mass *= 1e10 / CS.hubble_parameter

        if (mass < 11): return "No SNe"
        if (mass > 11 and mass <= 40): return "TypeII SNe"
        if (mass > 40 and mass <= 140): return "DCBH"
        if (mass > 140 and mass <= 260): return "PISN"
        if (mass > 260): return "DCBH"

    def histogram_bins(self, data, num, type="log"):
        if type == 'log':
            return np.logspace(np.log10(np.min(data)), np.log10(np.max(data)), num)
        if type == 'lin':
            return np.linspace(np.log10(np.min(data)), np.log10(np.max(data)), num)
    
    def unwrap(self, x):
        if isinstance(x, (list, tuple)) and len(x) == 1:
            return x[0]
        if hasattr(x, "size") and getattr(x, "size", None) == 1:
            return x.item()
        return x

    def EddingtonRate(self, M, e=0.1):
        MdotEd = 2.2e-8 * (0.1 / e) * M
        return MdotEd
    
    def EddingtonLuminosity(self, M):
        MdotEd = self.EddingtonRate(M)
        MdotEd *= CS.Msun_to_g / CS.yr_to_s
        EdLuminosity = 0.1 * MdotEd * pow(CS.c_in_cgs, 2)
        return EdLuminosity

    def EddingtonGrowth(self, M0, t, e=0.1):
        t_Edd = 4.5e7 * (e / 0.1)
        M = M0 * np.exp(t / t_Edd)
        return M

    def SuperEddingtonGrowth(self, M0, t, f_edd=1.0, e=0.1):
        t_Edd = 4.5e7 * (e / 0.1)
        M = M0 * np.exp(f_edd * t / t_Edd)
        return M

    def RadiationPressure(self, M, FEdd, mode=1, PhotonTrapping=False):
        L = self.EddingtonLuminosity(M)
        if not PhotonTrapping:
            if mode == 1: # Force feedback
                p = FEdd * L / CS.c_in_cgs
                return p
            if mode == 2: # Energy feedback
                E = FEdd * L
                return E
        else:
            if mode == 1:
                p = L * (1 + np.log(FEdd)) / CS.c_in_cgs
                return p
            if mode == 2:
                E = L * (1 + np.log(FEdd))
                return E
    
    def FindEddingtonFactor(self, M0, M, t, e=0.1):
        t_Edd = 4.5e7 * (e / 0.1)
        f_edd = (t_Edd / t) * np.log(M / M0)
        return f_edd

    def FindIMFUncerntainty(self):

        M_char = 20
        Mmin = 0.1
        Mmax = 300.0
        

        def xi_top(M):
            return M**(-1.3) * np.exp(-(M_char / M)**1.6)
        def xi_sal(M):
            return M**(-2.35)
        def xi_lar(M):
            return M**(-1.35)
        
        NTH, err1 = integrate.quad(xi_top, 40, 140, epsabs=0, epsrel=1e-5)
        print(f"NTH = {NTH:.2e}")
        MtotTH, err2 = integrate.quad(lambda m: m * xi_top(m), Mmin, Mmax, epsabs=0, epsrel=1e-5)

        etaTH = NTH / MtotTH

        NS1, _ = integrate.quad(xi_sal, 40, 140, epsabs=0, epsrel=1e-5)
        NS2, _ = integrate.quad(xi_sal, 260, Mmax, epsabs=0, epsrel=1e-5)

        NS = NS1 + NS2
        print(f"NS = {NS:.2e}, NS1 = {NS1:.2e}, NS2 = {NS2:.2e}")
        MtotS, _ = integrate.quad(xi_lar, Mmin, Mmax, epsabs=0, epsrel=1e-5)
        etaS = NS / MtotS

        R = etaTH / etaS
        print(f"Top-Heavy to Salpeter IMF ratio: {R:.2f}")
        return R
    
    def BondiHoyleAccretion(self, M, rho, cs, v): # in Msun, g/cm**3, km/s, km/s
        cs *= 1e5
        v *= 1e5
        BondiRadius = M * CS.Msun_to_g * CS.G_in_cgs / (cs**2 + v**2)
        Mdot = 4 * np.pi * rho * BondiRadius**2 * np.sqrt((1.12 * cs)**2 + v**2)
        Mdot = Mdot * CS.yr_to_s / CS.Msun_to_g
        return Mdot

    def FreeFallTimescale(self, mass, radius, mode=1): # in Msun, pc
        mass *= CS.Msun_to_g
        radius *= CS.pc_in_cgs
        if mode == 1: #Sphere infall
            density = 3 * mass / (4 * np.pi * radius**3)
            tff = np.sqrt(3 * np.pi / (32 * CS.G_in_cgs * density))
            return tff
        if mode == 2: #Point mass
            tff = np.pi / 2 * np.sqrt(radius**3 / (2 * CS.G_in_cgs * mass))
            return tff

    def PhotonTrappingLuminosity(self, FEdd, mode=1):
        if mode == 1:
            eta = 1 / 12
            return eta
        if mode == 2:
            eta = 1 - 4 / 3 * (1 / FEdd)**0.5
            return eta
        if mode == 3:
            eta = -1 + 2 / 3 * (1 / FEdd)**0.5 + 0.5 * (FEdd)**0.5
            return eta
        if mode == 4:
            A = pow(0.9963 - 0.9292 * 0.7, -0.5639)
            B = pow(4.627 - 4.445 * 0.7, -0.5524)
            C = pow(827.3 - 718.1 * 0.7, -0.7060)

            eta = A * (0.985/(1/FEdd + B) + 0.015/(1/FEdd + C))
            return eta
        
    def CoolingTimeSoundCrossingTimeRation(self, number_density, temperature, resolution, mu=0.6, f=0.13):
        ## Equation derived from Dalla Vecchia Schaye 2012

        ratio = 2.8e2 * pow(number_density, -1) * (temperature / 10**7.5) * pow(resolution / 100, -1) * pow(mu / 0.6, -3/2) * (f/0.13)

        return ratio

class ObservationData:
    def __init__(self):
        self.Mailino()
        self.Juodzbalis()
        self.Taylor()

    def Mailino(self):
        self.maiolino = {}
        self.maiolino[10013704] = {
            "MBH": [7.5, 0.3, 0.3],
            "MStar": [8.88, 0.66, 0.66],
            "f_edd": 0.06,
            "Redshift": 5.9193,
            "Lbol" : 44.29           
        }
        self.maiolino[8084] = {
            "MBH": [7.25, 0.3, 0.3],
            "MStar": [8.45, 0.03, 0.03],
            "f_edd": 0.16,
            "Redshift": 4.6482,
            "Lbol" : 44.25           
        }
        self.maiolino[1093] = {
            "MBH": [7.36, 0.3, 0.3],
            "MStar": [8.34, 0.2, 0.2],
            "f_edd": 0.2,
            "Redshift": 5.5951,
            "Lbol" : 44.32           
        }
        self.maiolino[3608] = {
            "MBH": [6.82, 0.3, 0.3],
            "MStar": [8.38, 0.11, 0.15],
            "f_edd": 0.11,
            "Redshift": 5.26894,
            "Lbol" : None           
        }
        self.maiolino[11836] = {
            "MBH": [7.12, 0.3, 0.3],
            "MStar": [7.79, 0.3, 0.3],
            "f_edd": 0.2,
            "Redshift": 4.40935,
            "Lbol" : 44.11           
        }
        self.maiolino[20621] = {
            "MBH": [7.3, 0.3, 0.3],
            "MStar": [8.06, 0.7, 0.7],
            "f_edd": 0.18,
            "Redshift": 4.68123,
            "Lbol" : 44.17           
        }
        self.maiolino[73488] = {
            "MBH": [7.71, 0.3, 0.3],
            "MStar": [9.78, 0.2, 0.2],
            "f_edd": 0.16,
            "Redshift": 4.1332,
            "Lbol" : 45.22          
        }
        self.maiolino[77652] = {
            "MBH": [6.86, 0.3, 0.3],
            "MStar": [7.87, 0.16, 0.28],
            "f_edd": 0.38,
            "Redshift": 5.22943,
            "Lbol" : 44.11           
        }
        self.maiolino[61888] = {
            "MBH": [7.22, 0.3, 0.3],
            "MStar": [8.11, 0.92, 0.92],
            "f_edd": 0.32,
            "Redshift": 5.87461,
            "Lbol" : 44.38           
        }
        self.maiolino[62309] = {
            "MBH": [6.56, 0.3, 0.3],
            "MStar": [8.12, 0.12, 0.13],
            "f_edd": 0.39,
            "Redshift": 5.17241,
            "Lbol" : 43.56           
        }
        self.maiolino[53757] = {
            "MBH": [7.69, 0.3, 0.3],
            "MStar": [10.18, 0.13, 0.12],
            "f_edd": 0.05,
            "Redshift": 4.4480,
            "Lbol" : 44.29           
        }
        self.maiolino[954] = {
            "MBH": [7.9, 0.3, 0.3],
            "MStar": [10.66, 0.09, 0.1],
            "f_edd": 0.42,
            "Redshift": 6.76026,
            "Lbol" : 45.17           
        }
        self.maiolino[11] = {
            "MBH": [6.2, 0.3, 0.3],
            "MStar": [8.9, 0.2, 0.3],
            "f_edd": 5.5,
            "Redshift": 10.6034,
            "Lbol" : 45           
        }
    
    def Juodzbalis(self):
        self.juodzbalis = {}
        self.juodzbalis[30148179] = {
            "MBH": [7.12, 0.3, 0.3],
            "MStar": [8.95, 0.58, 0.58],
            "f_edd": 0.11,
            "Redshift": 5.922,
            "Lbol" : 44.25           
        }
        self.juodzbalis[210600] = {
            "MBH": [7.41, 0.3, 0.3],
            "MStar": [8.40, 0.61, 0.61],
            "f_edd": 0.57,
            "Redshift": 6.306,
            "Lbol" : 44.29           
        }
        self.juodzbalis[209777] = {
            "MBH": [8.90, 0.3, 0.3],
            "MStar": [None, None, None],
            "f_edd": 0.27,
            "Redshift": 3.709,
            "Lbol" : 45.42           
        }
        self.juodzbalis[179198] = {
            "MBH": [7.23, 0.3, 0.3],
            "MStar": [8.44, 0.05, 0.05],
            "f_edd": 0.04,
            "Redshift": 3.830,
            "Lbol" : 43.92           
        }
        self.juodzbalis[172975] = {
            "MBH": [7.25, 0.3, 0.3],
            "MStar": [8.98, 0.14, 0.14],
            "f_edd": 0.15,
            "Redshift": 4.741,
            "Lbol" : 44.07           
        }
        self.juodzbalis[159717] = {
            "MBH": [7.44, 0.3, 0.3],
            "MStar": [None, None, None],
            "f_edd": 0.38,
            "Redshift": 5.077,
            "Lbol" : 45.13           
        }
        self.juodzbalis[159438] = {
            "MBH": [6.47, 0.3, 0.3],
            "MStar": [8.35, 0.13, 0.13],
            "f_edd": 0.35,
            "Redshift": 2.239,
            "Lbol" : 44.11           
        }
        self.juodzbalis[49729] = {
            "MBH": [7.67, 0.3, 0.3],
            "MStar": [None, None, None],
            "f_edd": 0.115,
            "Redshift": 3.189,
            "Lbol" : 44.83           
        }
        self.juodzbalis[38562] = {
            "MBH": [7.51, 0.3, 0.3],
            "MStar": [9.76, 0.09, 0.09],
            "f_edd": 0.12,
            "Redshift": 4.822,
            "Lbol" : 44.70           
        }
        self.juodzbalis[29648] = {
            "MBH": [6.81, 0.3, 0.3],
            "MStar": [9.71, 0.01, 0.01],
            "f_edd": 0.11,
            "Redshift": 2.960,
            "Lbol" : 43.90           
        }
        self.juodzbalis[17341] = {
            "MBH": [6.76, 0.3, 0.3],
            "MStar": [8.54, 0.03, 0.03],
            "f_edd": 0.15,
            "Redshift": 3.598,
            "Lbol" : 44.01           
        }
        self.juodzbalis[13329] = {
            "MBH": [6.86, 0.3, 0.3],
            "MStar": [9.52, 0.14, 0.14],
            "f_edd": 0.14,
            "Redshift": 3.936,
            "Lbol" : 44.11           
        }
        self.juodzbalis[9598] = {
            "MBH": [6.48, 0.3, 0.3],
            "MStar": [9.08, 0.38, 0.38],
            "f_edd": 0.18,
            "Redshift": 3.324,
            "Lbol" : 43.85           
        }
        self.juodzbalis[2916] = {
            "MBH": [7.05, 0.3, 0.3],
            "MStar": [8.98, 0.97, 0.97],
            "f_edd": 0.06,
            "Redshift": 3.664,
            "Lbol" : 43.91           
        }

    def Reines(self, mstellar):
        alpha, beta = 7.45, 1.05
        alpha_err, beta_err = 0.08, 0.11

        mbh = alpha + beta * np.log10(mstellar/1e11)
        mbh_err = mbh * np.sqrt((alpha_err / alpha)**2 + (beta_err / beta)**2 + (0.1 / np.log(10))**2)

        return mbh, mbh_err

    def Taylor(self):
        self.taylor = {}
        self.taylor[0] = {
            "MBH" : 6.25,
            "Phi" : [258, 125, 113]
        }
        self.taylor[1] = {
            "MBH" : 6.75,
            "Phi" : [276, 83.7, 59.8]
        }
        self.taylor[2] = {
            "MBH" : 7.25,
            "Phi" : [113, 32.5, 24.3]
        }
        self.taylor[3] = {
            "MBH" : 7.75,
            "Phi" : [36.1, 14.8, 10.5]
        }
        self.taylor[4] = {
            "MBH" : 8.25,
            "Phi" : [7.67, 7.51, 4.17]
        }



