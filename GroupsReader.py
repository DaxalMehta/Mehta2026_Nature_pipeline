import illustris_python as il
import numpy as np
import Utilities
from hmf import MassFunction
from astropy.cosmology import Planck15 as cosmo

CS = Utilities.Constants()

class Reader:
    def __init__(self, base, snap, halo_fields=None, subhalo_fields=None):
        self.base = base
        self.snap = snap
        self.halo_fields = ["GroupMass", "GroupPos", "Group_R_TopHat200", "GroupMassType"]
        if halo_fields:
            for field in halo_fields:
                self.subhalo_fields.append(field)
        self.subhalo_fields = ['SubhaloMass','SubhaloCM', 'SubhaloHalfmassRad', 'SubhaloGroupNr', 'SubhaloMassType']
        if subhalo_fields:
            for field in subhalo_fields:
                self.subhalo_fields.append(field)
        
        self.halos = self._ReadHalos()
        self.subhalos = self._ReadSubHalos()
        self.header = self._ReadHeader()
        self.Redshift = self.header['Redshift']

    def _ReadHeader(self):
        header = il.groupcat.loadHeader(self.base, self.snap)
        return header
    
    def _ReadHalos(self):
        halos = il.groupcat.loadHalos(self.base, self.snap, self.halo_fields)
        return halos
    
    def _ReadSubHalos(self):
        subhalos = il.groupcat.loadSubhalos(self.base, self.snap, self.subhalo_fields)
        return subhalos
    
    def HostHaloProperties(self, SinkPos):
        HaloPos = self.halos["GroupPos"]
        HaloRad = self.halos["Group_R_TopHat200"]
        HaloMassType = self.halos["GroupMassType"]
        dist = np.linalg.norm(HaloPos - SinkPos, axis=1)
        ind = np.argmin(dist)
        if (HaloMassType[ind, 5] == 0): 
            #print("Halo has no sink particles. The sink particle is an orphan.")
            return None, None
        else:
            return HaloPos[ind], HaloRad[ind]
    
    def HostSubhaloProperties(self, SinkPos):
        SubhaloPos = self.subhalos["SubhaloCM"]
        SubhaloRad = self.subhalos["SubhaloHalfmassRad"]
        SubhaloMassType = self.subhalos["SubhaloMassType"]
        dist = np.linalg.norm(SubhaloPos - SinkPos, axis=1)
        ind = np.argmin(dist)
        if (SubhaloMassType[ind, 5] == 0):
            #print("Subhalo has no sink particles. The sink particle is an orphan.")
            return None, None
        else:
            return SubhaloPos[ind], SubhaloRad[ind]
    
    def HostGalaxyProperties(self, SinkPos):
        SubhaloPos = self.subhalos["SubhaloCM"]
        SubhaloMass = self.subhalos["SubhaloMass"] * 1e10 / CS.hubble_parameter
        SubhaloRad = self.subhalos["SubhaloHalfmassRad"]
        SubhaloMassType = self.subhalos["SubhaloMassType"] * 1e10 / CS.hubble_parameter
        dist = np.linalg.norm(SubhaloPos - SinkPos, axis=1)
        print(np.min(dist), np.max(SubhaloRad))
        #ind = np.argsort(dist)
        mask = dist < SubhaloRad
        if len(SubhaloMass[mask]) < 1: return None, None
        SubhaloMass = SubhaloMass[mask]
        SubhaloRad = SubhaloRad[mask]
        SubhaloPos = SubhaloPos[mask]
        ind = np.argmax(SubhaloMass)
        return SubhaloPos[ind], SubhaloRad[ind]

        '''
        for i in range(10):
            if dist[ind[i]] < SubhaloRad[ind[i]]:
                if SubhaloMassType[ind[i], 5] > SinkMass:
                    return SubhaloPos[ind[i]], SubhaloRad[ind[i]]
        
        return None, None
        '''



    def Normalization(self, z, box_size=1.0):
        HaloMasses = self.halos["GroupMass"] * 1e10 
        SinkMasses = self.halos["GroupMassType"][:, 5] * 1e10
        
        mask = SinkMasses > 0
        mass_low = np.min(HaloMasses[mask])
        mass_high = np.max(HaloMasses[mask])

        mask = (HaloMasses >= mass_low) & (HaloMasses <= mass_high)
        Nhalo = mask.sum()
        if Nhalo == 0:
            print("No halos found in the specified mass range.")
            return None

        nhalo = Nhalo / box_size ** 3

        mf = MassFunction(cosmo_model=cosmo, z=z, Mmin=np.log10(mass_low)-1.0, Mmax=np.log10(mass_high)+1.0, dlog10m=0.01)

        #nhaloanalytic = nhaloanalytic = mf.integrate_hmf(mass_low, mass_high)
        mass = mf.m  # Msun/h (check units depending on settings)
        mask = (mass >= mass_low) & (mass <= mass_high)
        
        nhaloanalytic = np.trapz(mf.dndm[mask], mass[mask])

        norm = nhaloanalytic / nhalo
        print(f"N_halo (Sim) = {Nhalo}, N_halo (Analytic) = {nhaloanalytic:.2e}, Normalization = {norm:.2f}")
        return norm