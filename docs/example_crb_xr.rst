:orphan:

The Metagalactic X-ray Background
=================================
In this example, we'll compute the Meta-Galactic X-ray background over a
series of redshifts (:math:`10 \leq z \leq 40`):

::
    
    # Initialize radiation background
    pars = \
    {
     # Source properties
     'pop_sfr_model': 'sfrd-func',
     'pop_sfrd': lambda z: 0.1,
     
     'pop_sed': 'pl',
     'pop_alpha': -1.5,
     'pop_Emin': 2e2,
     'pop_Emax': 3e4,
     'pop_EminNorm': 5e2,
     'pop_EmaxNorm': 8e3,
     'pop_rad_yield': 2.6e39,
     'pop_rad_yield_units': 'erg/s/sfr',
     
     # Solution method
     'pop_solve_rte': True,
     'tau_redshift_bins': 400,

     'initial_redshift': 60.,
     'final_redshift': 5.,
    }
    
To summarize these inputs, we've got:

* A constant SFRD of :math:`0.1 \ M_{\odot} \ \mathrm{yr}^{-1} \ \mathrm{cMpc}^{-3}`, given by the ``pop_sfrd`` parameter.
* A power-law spectrum with index :math:`\alpha=-1.5`, given by ``pop_sed`` and ``pop_alpha``, extending from 0.2 keV to 30 keV.
* A yield of :math:`2.6 \times 10^{39} \ \mathrm{erg} \ \mathrm{s}^{-1} \ (M_{\odot} \ \mathrm{yr})^{-1}` in the :math:`0.5 \leq h\nu / \mathrm{keV} \leq  8` band, set by ``pop_EminNorm``, ``pop_EmaxNorm``, ``pop_yield``, and ``pop_yield_units``. This is the :math:`L_X-\mathrm{SFR}` relation found by `Mineo et al. (2012) <http://adsabs.harvard.edu/abs/2012MNRAS.419.2095M>`_.

See :doc:`params_populations` for a complete listing of parameters relevant to :class:`ares.populations.GalaxyPopulation` objects.
    
Now, to initialize a calculation:

::  

    import ares

    mgb = ares.simulations.MetaGalacticBackground(**pars)
    
Now, let's run the thing:

::

    mgb.run()
    
We'll pull out the evolution of the background just as we did in the previous  example:

::

    z, E, flux = mgb.get_history(flatten=True)

and plot up the result (at the final redshift):

::

    import matplotlib.pyplot as pl
    from ares.physics.Constants import erg_per_ev

    pl.semilogy(E, flux[0] * E * erg_per_ev, color='k')
    pl.xlabel(ares.util.labels['E'])
    pl.ylabel(ares.util.labels['flux_E'])
    
    z, E, flux = mgb.get_history(flatten=True)
                
Compare to the analytic solution, given by Equation A1 in `Mirocha (2014) <http://adsabs.harvard.edu/abs/2014arXiv1406.4120M>`_ (the *cosmologically-limited* solution to the radiative transfer equation)

.. math ::

    J_{\nu}(z) = \frac{c}{4\pi} \frac{\epsilon_{\nu}(z)}{H(z)} \frac{(1 + z)^{9/2-(\alpha + \beta)}}{\alpha+\beta-3/2} \times \left[(1 + z_i)^{\alpha+\beta-3/2} - (1 + z)^{\alpha+\beta-3/2}\right]

with :math:`\alpha = -1.5`, :math:`\beta = 0`, :math:`z=5`, and :math:`z_i=60`,

::

    import numpy as np
    from ares.physics.Constants import c, ev_per_hz    

    # Grab the GalaxyPopulation instance
    pop = mgb.pops[0] 

    # Compute cosmologically-limited solution
    e_nu = np.array(map(lambda E: pop.Emissivity(10., E), E))
    e_nu *= c / 4. / np.pi / pop.cosm.HubbleParameter(5.) 
    e_nu *= (1. + 5.)**6. / -3.
    e_nu *= ((1. + 60.)**-3. - (1. + 5.)**-3.)
    e_nu *= ev_per_hz

    # Plot it
    pl.semilogy(E, e_nu, color='b', ls='--')
    
Neutral Absorption by the Diffuse IGM
-------------------------------------   
The calculation above is basically identical to the optically-thin UV background calculations performed in the previous example, at least in the cases where we neglected any sawtooth effects. While there is no modification to the X-ray background due to resonant absorption in the Lyman series (of Hydrogen or Helium II), bound-free absorption by intergalactic hydrogen and helium atoms acts to harden the spectrum. By default, *ARES* will *not* include these effects.

To "turn on" bound-free absorption in the IGM, modify the dictionary of parameters you've got already:

::

    pars['tau_approx'] = 'neutral'

Now, initialize and run a new calculation:

::

    mgb2 = ares.simulations.MetaGalacticBackground(**pars)
    mgb2.run()
    
and plot the result on the same axes:

::

    z2, E2, flux2 = mgb2.get_history(flatten=True)

    pl.loglog(E2, flux2[0] * E2 * erg_per_ev, color='k', ls=':')
    
    pl.savefig('ares_crte_xr.png')

.. figure::  https://www.dropbox.com/s/gpl3n2c3r8gwmd3/ares_crte_xr.png?raw=1
   :align:   center
   :width:   600

   X-ray background spectrum, with (dotted) and without (solid) neutral absorption from the IGM. Analytic solution for optically-thin case in dashed blue.
    
    
The behavior at low photon energies (:math:`h\nu \lesssim 0.3 \ \mathrm{keV}`)
is an artifact that arises due to poor redshift resolution. This is a trade
made for speed in solving the cosmological radiative transfer equation,
discussed in detail in Section 3 of `Mirocha (2014)
<http://adsabs.harvard.edu/abs/2014arXiv1406.4120M>`_. For more accurate
calculations, you must enhance the redshift sampling using the ``tau_redshift_bins``
parameter, e.g.,

::

    pars['tau_redshift_bins'] = 500

The optical depth lookup tables that ship with *ARES* use ``tau_redshift_bins=400``
as a default. If you run with ``tau_redshift_bins=500``, you should see some improvement in the soft X-ray spectrum. It'll take a few minutes to generate a new table. Run ``$ARES/input/optical_depth/generate_optical_depth_tables.py`` to make more!

.. .. note :: Development of a dynamic optical depth calculation is underway, which can be turned on and off using the ``dynamic_tau`` parameter.

Alternative Methods
-------------------
The technique outlined above is the fastest way to integrate the cosmological radiative transfer equation (RTE), but it assumes that we can tabulate the optical depth ahead of time. What if instead we wanted to study the radiation background in a decreasingly opaque IGM? Well, we can solve the RTE at several photon energies in turn: ::

    E = np.logspace(2.5, 4.5, 100)
    
To determine the background intensity at :math:`z=10` due to the same BH population
as above, we could do something like: ::

    # Function describing evolution of IGM ionized fraction with respect to redshift
    # (fully ionized for all time in this case, meaning IGM is optically thin)
    xofz = lambda z: 1.0

    # Compute flux at z=10 and each observed energy due to emission from 
    # sources at 10 <= z <= 20.
    F = [rad.AngleAveragedFlux(10., nrg, zf=20., xavg=xofz) for nrg in E]

    pl.loglog(E, F)
    
You'll notice that computing the background intensity is much slower when
we do not pre-compute the IGM optical depth.    

Let's compare this to an IGM with evolving ionized fraction: :: 
    
    # Here's a function describing the ionization evolution for a scenario
    # in which reionization is halfway done at z=10 and somewhat extended.
    xofz2 = lambda z: ares.util.xHII_tanh(z, zr=10., dz=4.)
    
    # Compute fluxes
    F2 = [rad.AngleAveragedFlux(10., nrg, zf=20., xavg=xofz2) for nrg in E]
    
    # Plot results
    pl.loglog(E, F2)
    
    # Add some nice axes labels
    pl.xlabel(ares.util.labels['E'])
    pl.ylabel(ares.util.labels['flux'])    
    
Notice how the plot of ``F2`` has been hardened by neutral absorption in the IGM!
    
Self-Consistent Meta-Galactic Background & IGM
----------------------------------------------
If we don't already know the IGM optical depth *a-priori*, then the calculations above will only bracket the result expected in a more complex, evolving IGM, in which the radiation background ionizes the IGM, thus making the IGM more transparent, which then softens the meta-galactic background, and so on. A dynamic background calculator that takes this into account is on the *ARES* wish-list -- shoot me an email if you're so inclined.

