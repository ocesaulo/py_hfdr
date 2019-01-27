#!/usr/bin/env python3
'''
 Cythonized version of chirp handling functions and main loop.
 Author: Saulo M Soares
'''

import numpy as np
from scipy import signal
import ctypes as ct
cimport numpy as np
cimport cython


def chirp_compress(double[:, :] chirp_in, int compression_factor):
    '''window, decimate and unapply window to chirp'''

    cdef int d = np.max(chirp_in.shape)
    cdef double[:] w = signal.windows.blackmanharris(d).T
    cdef double w1
    cdef double wc
    cdef double dc
    w1 = signal.windows.blackmanharris(
                                       np.int(np.ceil(d / compression_factor))
                                       ).T
    wc = np.asarray(chirp_in) * np.asarray(w)
    dc = signal.decimate(wc, compression_factor)
    return dc / w1


def chirp_prep(double[:, :] chirp_in, int len_end, int SHIFT, int SHIFT_POS):
    '''edit chirp'''

    cdef int d = np.max(chirp_in.shape)
    cdef int start_spot = (d - len_end) // 2
    cdef int end_spot = start_spot + len_end
    cdef short[:, :] chirp_int = np.asrray(chirp_in) / (2**SHIFT)
    return chirp_int[start_spot:end_spot]


def loop_chirps(fi, fo, site_conf):
    '''Does the main loop for the chirp handling from reading file.'''

    cdef int ichirp, ichan
    cdef int IQ = site_conf.vars.IQ
    cdef int MTC = site_conf.vars.MTC
    cdef int MTCL = site_conf.vars.MTCL
    cdef int NANT = site_conf.vars.NANT
    cdef int NCHIRP = site_conf.vars.NCHIRP
    cdef int MT = site_conf.vars.MT
    cdef int MTL = site_conf.vars.MTL
    cdef int NCHAN = site_conf.vars.NCHAN
    cdef int OVER = site_conf.vars.OVER
    cdef int SHIFT_POS = site_conf.vars.SHIFT_POS
    cdef int SHIFT = site_conf.vars.SHIFT
    cdef int COMP_FAC = site_conf.vars.COMP_FAC
    cdef int[:, :, :] map = site_conf.map

    cdef short[:, :, :, :] wera = np.zeros((IQ, MTC, NANT, NCHIRP), dtype=ct.c_int16)

    cdef int i0, i2
    cdef int indata
    cdef double[:, :] data
    cdef double[:, :] sdata
    cdef double[:, :] datac
    cdef short[:, :, :] wera1
    cdef short[:, :, :] werac

    for ichirp in range(0, NCHIRP):

        # initialize variables per chirp
        sdata = np.zeros((NCHAN * IQ, MTL), dtype=ct.c_float64)
        datac = np.zeros((NCHAN * IQ, MTCL), dtype=ct.c_float64)
        wera1 = np.zeros((IQ, MT, NANT), dtype=ct.c_int16)
        werac = np.zeros((IQ, MTC, NANT), dtype=ct.c_int16)

        #  move back a bit in file to extend chirp so filter works cleanly
        fi.seek(-4 * NCHAN * IQ * SHIFT_POS, 1)
        indata = np.fromfile(fi, ct.c_int32, NCHAN * IQ * (MT * OVER + SHIFT_POS))
        data = indata.reshape((NCHAN * IQ, MT * OVER + SHIFT_POS)).astype(ct.c_float64)

        # manipulate data for each channel
        for ichan in range(0, NANT * IQ):
            # window, decimate, and unwindow
            sdata[ichan, :] = chirp_compress(data[ichan, :], OVER)
            datac[ichan, :] = chirp_compress(sdata[ichan, :], COMP_FAC)
            # reorder channels, shift bits, and move to int16 data
            i0 = np.intc(map[ichan, 2])
            i2 = np.intc(map[ichan, 1])
            wera1[i0, :, i2] = chirp_prep(sdata[ichan, :], MT, SHIFT, SHIFT_POS)
            werac[i0, :, i2] = chirp_prep(datac[ichan, :], MTC, SHIFT, SHIFT_POS)
        wera[..., ichirp] = werac  # store compressed data in 'wera'
        fo.write(wera1)  # write out data to RAW bin output file
    return wera
