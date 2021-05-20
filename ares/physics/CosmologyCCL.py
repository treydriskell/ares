"""

CosmologyCCL.py

Author: Jordan Mirocha
Affiliation: McGill University
Created on: Sat 15 Aug 2020 14:59:44 EDT

Description:

"""

import numpy as np
from .Cosmology import CosmologyARES
from .Constants import c, G, km_per_mpc, m_H, m_He, sigma_SB, g_per_msun, \
    cm_per_mpc, cm_per_kpc, k_B, m_p

try:
    import pyccl
except ImportError:
    raise
    pass

class CosmologyCCL(CosmologyARES):
    """
    Create a class instance that looks like a CosmologyARES instance but is
    calling CCL under the hood.
    """

    @property
    def _ccl_instance(self):
        if not hasattr(self, '_ccl_instance_'):
            ccl_kwargs = dict(Omega_c=self.omega_cdm_0,
                                Omega_b=self.omega_b_0, h=self.h70, 
                                n_s=self.primordial_index,
                                sigma8=self.sigma_8,)


            if self.pf['cosmology_helper'] is None:
                cosmo = pyccl.Cosmology(**ccl_kwargs, transfer_function='boltzmann_camb')

            # Set background quantities in CCL using class arrays, if cosmology_helper is passed
            # CCL commit 593ed60c required to make this work.
            else:
                cl = self.pf['cosmology_helper']
                bg = cl.get_background()

                a_bg = 1/(1+bg['z'])
                # Distances
                chi = bg['comov. dist.']
                # Expansion rate
                h_over_h0 = bg['H [1/Mpc]']
                h_over_h0 /= h_over_h0[-1]

                # Growth
                growth_factor = bg['gr.fac. D']
                growth_rate = bg['gr.fac. f']


                # Power spectra
                k_arr = np.logspace(-5, np.log10(self.pf['kmax']), 1000)
                nk = len(k_arr)

                z_pk = np.arange(self.pf['hmf_zmin'], self.pf['hmf_zmax'], self.pf['hmf_dz'])
                a_arr = 1 / (1 + z_pk[::-1])

                # Linear
                pkln = np.array([[cl.pk_lin(k, 1./a-1)
                                  for k in k_arr]
                                 for a in a_arr])
                # non-linear
                pknl = np.array([[cl.pk(k, 1./a-1)
                                  for k in k_arr]
                                 for a in a_arr])

                cosmo = pyccl.CosmologyCalculator(**ccl_kwargs,
                                    background={'a': a_bg, 'chi': chi, 'h_over_h0': h_over_h0},
                                    growth={'a': a_bg, 'growth_factor': growth_factor,
                                            'growth_rate': growth_rate},
                                    pk_linear={'a': a_arr, 'k': k_arr,
                                               'delta_matter:delta_matter': pkln},
                                    pk_nonlin={'a': a_arr, 'k': k_arr,
                                               'delta_matter:delta_matter': pknl})

                # # erase classy object for serialization purposes?
                # self.pf['cosmology_helper'] = 'used'

            self._ccl_instance_ = cosmo


            #'hmf_dlna': 2e-6,           # hmf default value is 1e-2
            #'hmf_dlnk': 1e-2,
            #'hmf_lnk_min': -20.,
            #'hmf_lnk_max': 10.,
            #'hmf_transfer_k_per_logint': 11,
            #'hmf_transfer_kmax': 100.,

            self._ccl_instance_.cosmo.gsl_params.INTEGRATION_EPSREL = 1e-8
            self._ccl_instance_.cosmo.gsl_params.INTEGRATION_DISTANCE_EPSREL = 1e-5
            self._ccl_instance_.cosmo.gsl_params.INTEGRATION_SIGMAR_EPSREL = 1e-12
            self._ccl_instance_.cosmo.gsl_params.ODE_GROWTH_EPSREL = 1e-8
            self._ccl_instance_.cosmo.gsl_params.EPS_SCALEFAC_GROWTH = 1e-8

            # User responsible for making sure NM and DELTA are consistent.
            self._ccl_instance.cosmo.spline_params.LOGM_SPLINE_MIN = 4
            self._ccl_instance.cosmo.spline_params.LOGM_SPLINE_MAX = 18
            self._ccl_instance.cosmo.spline_params.LOGM_SPLINE_NM = 1400
            self._ccl_instance.cosmo.spline_params.LOGM_SPLINE_DELTA = 0.01

            self._ccl_instance_.cosmo.spline_params.K_MIN = 1e-5
            self._ccl_instance_.cosmo.spline_params.K_MAX = float(self.pf['kmax'])
            self._ccl_instance_.cosmo.spline_params.K_MAX_SPLINE = float(self.pf['kmax'])
            self._ccl_instance_.cosmo.spline_params.N_K = 1000

            self._ccl_instance.cosmo.spline_params.A_SPLINE_NA = 500
            #self._ccl_instance.cosmo.spline_params.A_SPLINE_MIN_PK = 0.01



        return self._ccl_instance_

    def MeanMatterDensity(self, z):
        return pyccl.rho_x(self._ccl_instance, 1./(1.+z), 'matter') * g_per_msun / cm_per_mpc**3

    def MeanBaryonDensity(self, z):
        return (self.omega_b_0 / self.omega_m_0) * self.MeanMatterDensity(z)

    def MeanHydrogenNumberDensity(self, z):
        return (1. - self.Y) * self.MeanBaryonDensity(z) / m_H

    def MeanHeliumNumberDensity(self, z):
        return self.Y * self.MeanBaryonDensity(z) / m_He

    def MeanBaryonNumberDensity(self, z):
        return self.MeanBaryonDensity(z) / (m_H * self.MeanHydrogenNumberDensity(z) +
            4. * m_H * self.y * self.MeanHeliumNumberDensity(z))

    def ComovingRadialDistance(self, z0, z):
        """
        Return comoving distance between redshift z0 and z, z0 < z.
        """

        d0 = pyccl.comoving_radial_distance(self._ccl_instance, 1./(1.+z)) \
            * cm_per_mpc

        if z0 == 0:
            return d0

        d1 = pyccl.comoving_radial_distance(self._ccl_instance, 1./(1.+z0)) \
            * cm_per_mpc

        return d0 - d1

    def ProperRadialDistance(self, z0, z):
        return self.ComovingRadialDistance(z0, z) / (1. + z0)

    def dldz(self, z):
        """ Proper differential line element. """
        return self.ProperLineElement(z)

    def LuminosityDistance(self, z):
        return pyccl.luminosity_distance(self._ccl_instance, 1./(1.+z))
