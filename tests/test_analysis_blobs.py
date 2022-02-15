"""

test_analysis_blobs.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Thu May 26 11:28:43 PDT 2016

Description:

"""

import os
import ares
import numpy as np
from ares.util.Pickling import write_pickle_file

def test(Ns=500, Nd=4, prefix='test'):

    # Step 1. Make some fake data.

    # Start with a 2-D array that looks like an MCMC chain with 500 samples in a
    # 4-D parameter space. It's flat already (i.e., no walkers dimension)
    chain = np.reshape(np.random.normal(loc=0, scale=1., size=Ns*Nd), (Ns, Nd))

    # Random "likelihoods" -- just a 1-D array
    logL = np.random.rand(Ns)

    # Info about the parameters
    pars = ['par_{}'.format(i) for i in range(Nd)]
    is_log = [False] * Nd
    pinfo = pars, is_log

    # Write to disk.
    write_pickle_file(chain, '{!s}.chain.pkl'.format(prefix), ndumps=1,\
        open_mode='w', safe_mode=False, verbose=False)
    write_pickle_file(pinfo, '{!s}.pinfo.pkl'.format(prefix), ndumps=1,\
        open_mode='w', safe_mode=False, verbose=False)
    write_pickle_file(logL, '{!s}.logL.pkl'.format(prefix), ndumps=1,\
        open_mode='w', safe_mode=False, verbose=False)

    # Make some blobs. 0-D, 1-D, and 2-D.
    setup = \
    {
     'blob_names': [['blob_0'], ['blob_1'], ['blob_2', 'blob_3']],
     'blob_ivars': [None, [('x', np.arange(10))],
        [('x', np.arange(10)), ('y', np.arange(10,20))]],
     'blob_funcs': None,
    }

    write_pickle_file(setup, '{!s}.setup.pkl'.format(prefix), ndumps=1,\
        open_mode='w', safe_mode=False, verbose=False)

    # Blobs
    blobs = {}
    for i, blob_grp in enumerate(setup['blob_names']):

        if setup['blob_ivars'][i] is None:
            nd = 0
        else:
            # ivar names, ivar values
            ivn, ivv = list(zip(*setup['blob_ivars'][i]))
            nd = len(np.array(ivv).squeeze().shape)

        for blob in blob_grp:

            dims = [Ns]
            if nd > 0:
                dims.extend(list(map(len, ivv)))

            size = np.product(dims)
            data = np.reshape(np.random.normal(size=size), dims)

            # Add some nans to blob_3 to test our handling of them
            if blob == 'blob_3':
                num = int(np.product(dims) / 10.)

                mask_inf = np.ones(size)
                r = np.unique(np.random.randint(0, size, size=num))
                mask_inf[r] = np.inf

                data *= np.reshape(mask_inf, dims)

            write_pickle_file(data,\
                '{0!s}.blob_{1}d.{2!s}.pkl'.format(prefix, nd, blob),\
                ndumps=1, open_mode='w', safe_mode=False, verbose=False)

    # Now, read stuff back in and make sure ExtractData works. Plotting routines?
    anl = ares.analysis.ModelSet(prefix)

    # Test a few things.

    # Test data extraction
    for par in anl.parameters:
        data = anl.ExtractData(par)

    # Second, finding error-bars.
    for par in anl.parameters:
        mu, bounds = anl.get_1d_error(par, nu=0.68)

    # Test blobs, including error-bars, extraction, and plotting.
    for blob in anl.all_blob_names:
        data = anl.ExtractData(blob)

    for i, par in enumerate(anl.all_blob_names):

        # Must distill down to a single number to use this
        grp, l, nd, dims = anl.blob_info(par)

        if nd > 0:
            ivars = anl.blob_ivars[grp][l]
            slc = np.zeros_like(dims)
            iv = ivars[slc]
        else:
            iv = None

        mu, bounds = anl.get_1d_error(par, ivar=iv, nu=0.68)

    # Plot test: first, determine ivars and then plot blobs against eachother.
    ivars = []
    for i, blob_grp in enumerate(setup['blob_ivars']):
        if setup['blob_ivars'][i] is None:
            nd = 0
        else:
            ivn, ivv = list(zip(*setup['blob_ivars'][i]))
            nd = len(np.array(ivv).squeeze().shape)

        if nd == 0:
            ivars.append(None)
        elif nd == 1:
            ivn, ivv = list(zip(*setup['blob_ivars'][i]))
            ivars.append(ivv[0][0])
        else:
            ivn, ivv = list(zip(*setup['blob_ivars'][i]))
            ivars.append([ivv[0][0], ivv[1][0]])

    # Last set of ivars (remember: there are 2 2-D blobs)
    ivars.append([ivv[0][1], ivv[1][1]])
        
    # Cleanup
    for suffix in ['chain', 'logL', 'pinfo', 'setup']:
        os.remove('{0!s}.{1!s}.pkl'.format(prefix, suffix))

    for i, blob_grp in enumerate(setup['blob_names']):

        if setup['blob_ivars'][i] is None:
            nd = 0
        else:
            ivn, ivv = list(zip(*setup['blob_ivars'][i]))
            nd = len(np.array(ivv).squeeze().shape)

        for blob in blob_grp:
            os.remove('{0!s}.blob_{1}d.{2!s}.pkl'.format(prefix, nd, blob))

    assert True

if __name__ == '__main__':
    test()
