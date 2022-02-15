"""

test_const_ionization.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Oct 16 14:46:48 MDT 2014

Description:

"""

import ares
import numpy as np
from ares.physics.CrossSections import PhotoIonizationCrossSection as sigma

s_per_yr = ares.physics.Constants.s_per_yr

pars = \
{
 'problem_type': 0,
 'grid_cells': 1,
 'initial_ionization': [1.-1e-6, 1e-6],
 #'initial_temperature': 1e4,# make cold so collisional ionization is negligible
 'isothermal': False,

 'stop_time': 10.0,
 'plane_parallel': True,
 'recombination': False,  # To match analytical solution

 'source_type': 'toy',
 'source_qdot': 1e4,      # solver fails when this is large (like 1e10)
 'source_lifetime': 1e10,
 'source_E': [13.60000001],
 'source_LE': [1.0],
 'secondary_ionization': 0,
 'collisional_ionization': 0,
 'logdtDataDump': 0.5,
 'initial_timestep': 1e-15,
}

def test(rtol=1e-2):

    # Numerical solution
    sim = ares.simulations.RaySegment(**pars)
    sim.run()

    t, xHII = sim.get_cell_evolution(field='h_2')

    # Analytic solution: exponential time evolution
    sigma0 = sigma(pars['source_E'][0])
    qdot = pars['source_qdot']
    Gamma = qdot * sigma0

    xi0 = pars['initial_ionization'][1]
    C = 1. - xi0
    def xi(t, Gamma=Gamma):
        return 1. - C * np.exp(-Gamma * t)

    xHII_anyl = np.array(list(map(xi, t)))

    # Only test accuracy at somewhat later times
    mask = t > 0

    err = np.abs(xHII[mask] - xHII_anyl[mask]) / xHII_anyl[mask]

    assert np.allclose(xHII[mask], xHII_anyl[mask], rtol=rtol, atol=0)

if __name__ == '__main__':
    test()
