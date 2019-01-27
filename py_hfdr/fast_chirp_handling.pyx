#!/usr/bin/env python3
'''
 Cythonized version of chirp handling functions and main loop.
 Author: Saulo M Soares
'''

import numpy as np
from scipy import signal
# import ctypes as ct
cimport numpy as np
cimport cython


def chirp_compress(np.ndarray[np.double_t, ndim=1] chirp_in, int compression_factor):
    '''window, decimate and unapply window to chirp'''

    cdef int d = np.max((chirp_in.shape[0], chirp_in.shape[1]))
    cdef np.ndarray[np.double_t, ndim=1] w = signal.windows.blackmanharris(d).T
    cdef np.ndarray[np.double_t, ndim=1] w1
    cdef np.ndarray[np.double_t, ndim=1] wc
    cdef np.ndarray[np.double_t, ndim=1] dc
    w1 = signal.windows.blackmanharris(
                                       np.int(np.ceil(d / compression_factor))
                                       ).T
    wc = chirp_in * w
    dc = signal.decimate(wc, compression_factor)
    return dc / w1


def chirp_prep(np.ndarray[np.double_t, ndim=1] chirp_in, int len_end, int SHIFT, int SHIFT_POS):
    '''edit chirp'''

    cdef int d = np.max((chirp_in.shape[0], chirp_in.shape[1]))
    cdef int start_spot = (d - len_end) // 2
    cdef int end_spot = start_spot + len_end
    cdef np.ndarray[np.int16_t, ndim=1] chirp_int = np.int16(chirp_in / (2**SHIFT))
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
    cdef long[:, :] map = site_conf.MAP

    cdef np.ndarray[np.int16_t, ndim=4] wera = np.zeros((IQ, MTC, NANT, NCHIRP), dtype=np.short)

    cdef int i0, i2
    cdef int[:] indata
    cdef np.ndarray[np.float64_t, ndim=2] data
    cdef np.ndarray[np.float64_t, ndim=2] sdata
    cdef np.ndarray[np.float64_t, ndim=2] datac
    cdef np.ndarray[np.int16_t, ndim=3] wera1
    cdef np.ndarray[np.int16_t, ndim=3] werac

    for ichirp in range(0, NCHIRP):

        # initialize variables per chirp
        sdata = np.zeros((NCHAN * IQ, MTL))
        datac = np.zeros((NCHAN * IQ, MTCL))
        wera1 = np.zeros((IQ, MT, NANT), dtype=np.short)
        werac = np.zeros((IQ, MTC, NANT), dtype=np.short)

        #  move back a bit in file to extend chirp so filter works cleanly
        fi.seek(-4 * NCHAN * IQ * SHIFT_POS, 1)
        indata = np.fromfile(fi, np.int32, NCHAN * IQ * (MT * OVER + SHIFT_POS))
        # data = indata.reshape((NCHAN * IQ, MT * OVER + SHIFT_POS)).astype(ct.c_double)
        data = np.reshape(np.float64(indata), (NCHAN * IQ, MT * OVER + SHIFT_POS))

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
