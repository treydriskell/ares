:orphan:

Population Parameters
=====================

Basic Properties
----------------
``pop_zform``
    Redshift when sources "turn on."

    Default: 50

``pop_zdead``
    Population will not contribute to radiation backgrounds after this redshift.

    Default: 0

``pop_lw_src`` 
    Sources contribute to Lyman-Werner background? Only relevant if LW feedback is turned on.

    Default: ``True``
    
``pop_lya_src`` 
    Sources contribute to Ly-:math:`\alpha` background?
    
    Default: ``True``

``pop_ion_src_cgm`` 
    Sources contribute to growth of HII regions?

    Default: ``True``

``pop_ion_src_igm`` 
    Sources contribute ionization in bulk IGM?
    
    If ``approx_xrb=True``, this ionization rate assumes a mean X-ray photon energy
    of ``xray_Eavg``, which is 500 eV by default.

    Default: ``True``
    
``pop_heat_src_igm``
    Sources emit X-rays and heat bulk IGM?
    
    Default: ``True``
    
``pop_solve_rte``
    Solve the cosmological radiative transfer equation (RTE) in detail?
    
    Options: bool, list, tuple
    
    Default: ``False``
    
Star formation history
----------------------    
The following parameters control the star-formation history of a population. See :doc:`uth_pop_sfrd` for more information.

``pop_sfr_model``
    Value determines how star-formation history is computed.
    
    Options:
        + ``fcoll``: Relate SFRD to rate of collapse onto halos above minimum virial temperature (``pop_Tmin``) or mass (``pop_Mmin``) threshold assuming constant efficiency of star formation (``pop_fstar``).
        + ``sfe-func``: Model star formation efficiency as function halo mass and (perhaps) redshift. See next section for more details.
        + ``sfrd-func``: User-supplied function of redshift. See ``pop_sfrd`` below.
        + ``link:<ID>``: Link the SFRD to population with the given ``ID`` number (in <>'s).
    
``pop_Tmin``
    Minimum virial temperature of star-forming halos.
    
    Default: :math:`10^4` [Kelvin]
    
``pop_Mmin``
    Minimum mass of star-forming halos. Will override ``Tmin`` if set to 
    something other than ``None``.

    Default: ``None`` [:math:`M_{\odot}`]

``pop_fstar``
    Star formation efficiency, :math:`f_{\ast}`, i.e., fraction of collapsing
    gas that turns into stars.
    
    Options:
        + Any number between 0 and 1.
        + ``pq`` for "parameterized halo property". Requires ``pop_model=True``. See next section for more details on setting the ``pq_*`` parameters. Note that if multiple PQ's are being used for a single population, you can use square brackets to attach an ID number, e.g., ``pop_fstar=pq[0]`` and ``pop_fesc=pq[1]``. The square brackets w/ ID numbers must be appended to each of the corresponding ``pq_*`` parameters as well.
    
    Default: 0.1
    
    .. note :: If you set ``fstar`` to ``None``, the strength of radiation 
        backgrounds will be determined by the :math:`\xi` parameters, 
        ``xi_LW``, ``xi_XR``, and ``xi_UV``.

``pop_sfrd``
    The star formation rate density (SFRD) as a function of redshift. If provided, will override ``Tmin`` and ``Mmin``. For example, a constant (co-moving) SFRD of :math:`1 \ M_{\odot} \ \text{yr}^{-1} \ \text{cMpc}^{-3}` would be ``sfrd=lambda z: 1.0``. Also must set ``pop_sfrd_units='msun/yr/mpc^3' (see below).
    
    Default: ``None`` 
        
``pop_sfrd_units``
    Sets the units of the parametric form for the SFRD (``pop_sfrd``).
    
    Options:
        + ``msun/yr/mpc^3`` for :math:`M_{\odot} \ \text{yr}^{-1} \ \text{cMpc}^{-3}`
        + ``g/s/cm^3``
    
    Default: ``g/s/cm^3``
    
``pop_calib_lum``
    If not ``None``, this parameter will guarantee that the :math:`1600\unicode{x212B}` ( by default) luminosity (per unit star formation) is fixed at the provided value. This can be useful if, for example, you're modeling the galaxy luminosity function (LF) and want to change the stellar population model while preserving the LF. See Section 3.4 of `Mirocha, Furlanetto, & Sun (2017) <http://adsabs.harvard.edu/abs/2017MNRAS.464.1365M>`_ for further discussion of this.

``pop_calib_wave``
	Wavelength to which ``pop_calib_lum`` refers. 
	
	Default: :math:`1600 \ \unicode{x212B}`
    
Radiation Fields
----------------
``pop_sed_model``
    Treat the SED of this source population in detail? If `True`, it means that we use parameters like ``pop_sed``, ``pop_Emin``, ``pop_Emax``, etc. in order to set the overall normalization of the emission. If False, parameters like ``pop_Nlw``, ``pop_Nion``, and ``pop_fX`` are used instead of ``pop_rad_yield``.

    See :doc:`uth_pop_radiation` for more information.

    Default: ``True``

``pop_rad_yield``
    How many photons are emitted per unit star formation?

    Default: :math:`2.6 \times 10^{39}`

``pop_rad_yield_units``
    How to normalize the yield? 

    Options: 

    + ``erg/s/SFR`` [i.e., :math:`\mathrm{erg} \ \mathrm{s}^{-1} \ (M_{\odot} \ \mathrm{yr}^{-1})^{-1}`]
    + ``photons/baryon``
    + ``photons/Msun``

    Default: ``erg/s/SFR``

Internally, all units are cgs, which means at run-time all yields will be converted to units of :math:`\mathrm{erg} \ \mathrm{g}^{-1}`.

These parameters of course dictate an amount of energy produced per unit star formation *in a particular band*. That band is specified by the ``pop_EminNorm`` and ``pop_EmaxNorm`` parameters.

``pop_EminNorm``
    Minimum photon energy to consider in normalization.

    Default: 200 [eV]

``pop_EmaxNorm``
    Maximum photon energy to consider in normalization.

    Default: 3e4 [eV]

To be precise,

.. math ::

    \int_{\texttt{pop_EminNorm}}^{\texttt{pop_EmaxNorm}} \frac{\epsilon_{\nu}}{\dot{\rho}_{\ast}} d\nu = \frac{\texttt{pop_yield}}{\texttt{pop_yield_units}}

where :math:`\epsilon_{\nu}` is the emissivity of the population and :math:`\dot{\rho}_{\ast}` is the star-formation rate density (SFRD).

This range does not necessarily determine the band in which photons are emitted. For example, you might want to normalize the emission in the 0.5-8 keV band (e.g., if you're adopting the :math:`L_X`-SFR relation), but allow sources to emit at all energies. To do so, you must choose an SED, which then gets used to extrapolate the 0.5-8 keV yield to lower/higher energies.

We use square brackets on this page to denote the units of parameters.

``pop_sed``
    Spectral energy distribution assumed for this population.

    Options:

    + ``'bb'``: blackbody. If supplied, ``pop_temperature`` sets assumed blackbody temperature.
    + ``'pl'``: power-law. If supplied, ``pop_alpha`` parameter sets power-law index.
    + ``'mcd'``; Multi-color disk (Mitsuda et al. 1984)
    + ``'simpl'``: SIMPL Comptonization model (Steiner et al. 2009)
    + ``'qso'``: Quasar template spectrum (Sazonov et al. 2004)
    + ``leitherer1999``: Stellar population synthesis models from the original `starburst99 <http://www.stsci.edu/science/starburst99/docs/default.htm>`_ dataset.
    + ``eldridge2009``: Stellar population synthesis models from `BPASS <http://bpass.auckland.ac.nz/>`_ version 1.0 models.

``pop_Z``
    If ``pop_sed`` is ``leitherer1999`` or ``eldridge2009``, this is the stellar metallicity assumed for the synthesis models. Can take on values in the range :math:`0.001 \leq Z \leq 0.04``.

    Default: 0.02 (solar)
    
``pop_imf``
    If ``pop_sed`` is ``leitherer1999`` or ``eldridge2009``, this is the stellar initial mass function used.

    Default: 2.35 (Salpeter)    
    
``pop_nebular``
    Whether or not to include nebular emission.
    
    Default: ``False``

``pop_ssp``
    Whether or not to assume a "simple stellar population," i.e., an instantaneous burst of star formation. If ``False``, assumes continuous star formation.
    
    Default: ``False``

``pop_binaries``
    If ``pop_sed`` is ``eldridge2009``, this dictates whether binary systems are included in the model.
    
    Default: ``False``

``pop_Emin``
    Minimum photon energy to consider in radiative transfer calculation.

    Default: 200 [eV]

``pop_Emax``
    Maximum photon energy to consider in radiative transfer calculation. 

    Default: 3e4 [eV]

            
For backward compatibility
--------------------------
There are many parameters that do *not* have the ``pop_`` prefix attached to them, but are nonetheless convenient because they are the most common parameters in fiducial global 21-cm models. In addition, in *ARES* version 0.1, the ``pop_`` formulation was not yet in place, and the following parameters were the norm. They can still be used for ``problem_type=101`` (see :doc:`problem_types`), but one should be careful otherwise.

``cX``
    Normalization of the X-ray luminosity to star formation rate (:math:`L_X`-SFR) relation in 
    band given by ``pop_EminNorm`` and ``pop_EmaxNorm``. If ``approx_xrb=1``, this represents the X-ray luminosity density per unit star formation, such that the heating
    rate density will be equal to :math:`\epsilon_X = f_{X,h} c_X f_X \times \text{SFR}`.

    Default: :math:`3.4 \times 10^{40}` [:math:`\text{erg} \ \text{s}^{-1} \ (M_{\odot} \ \mathrm{yr}^{-1})^{-1}`]
    
``fX``
    Constant multiplicative factor applied to ``cX``, which is typically 
    chosen to match observations of nearby star-forming galaxies, i.e., 
    ``fX`` parameterizes ignorance in redshift evolution of ``cX``.
    
    Default: 0.2

``Nlw``
    Number of photons emitted in the Lyman-Werner band per baryon of star formation.
    
    If ``fstar`` is *not* ``None``, the co-moving LW luminosity density is given by :math:`f_{\ast} N_{\mathrm{LW}} \text{SFRD}`.
    
    Default: 9690
    
``Nion``
    Number of ionizing photons emitted per baryon of star formation.
    
    Default: 4000
    
``fesc``
    Escape fraction of ionizing radiation.
    
    Default: 0.1

``xi_UV``
    Ionizing efficiency, :math:`\xi_{\mathrm{UV}}`. If supplied, overrides ``fesc``, ``Nion``, and ``fstar``, as it is defined by:
        
        :math:`\xi_{\mathrm{UV}} \equiv f_{\ast} f_{\mathrm{esc}} N_{\mathrm{ion}}`

    Default: `None`

``xi_LW``
    Lyman-Werner efficiency, :math:`\xi_{\mathrm{LW}}`. If supplied, overrides ``Nlw``, and ``fstar``, as it is defined by:

        :math:`\xi_{\mathrm{LW}} \equiv f_{\ast} N_{\mathrm{LW}}`

    Default: `None`


``xi_XR``
    X-ray efficiency, :math:`\xi_{\mathrm{XR}}`. If supplied, overrides  ``fX`` and ``fstar``, as it is defined by:

        :math:`\xi_{\mathrm{XR}} \equiv f_{\ast} f_X`

    Default: `None`
