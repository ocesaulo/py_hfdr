'''
 Cythonized version of chirp handling functions and main loop.
 Need check if actually optimized... maybe array index is bad
 But profiling indicates that the scipy calls are the bottleneck
 Author: Saulo M Soares
'''

import numpy as np
from scipy.signal import decimate
from scipy.signal.windows import blackmanharris
# import ctypes as ct
cimport numpy as np
cimport cython


@cython.boundscheck(False)  # turn off bounds-checking for entire function
# @cython.wraparound(False)  # turn off negative index wrapping for entire func
def chirp_compress(double[:] chirp_in, int compression_factor):
    '''window, decimate and unapply window to chirp'''

    cdef int n
    cdef int d = np.max((chirp_in.shape[0], chirp_in.shape[1]))
    cdef double[:] w = blackmanharris(d)
    cdef double[:] w1 = blackmanharris(np.int(np.ceil(d / compression_factor)))
    cdef double[:] wc = np.zeros((len(chirp_in),))

    # wc = chirp_in * w
    for n in range(len(chirp_in)):
        wc[n] = chirp_in[n] * w[n]
    cdef double[:] dc = decimate(wc, compression_factor)
    cdef double[:] result = np.zeros((len(dc),))
    for n in range(len(dc)):
        result[n] = dc[n] / w1[n]
    # return dc / w1
    return result.base


@cython.boundscheck(False)  # turn off bounds-checking for entire function
# @cython.wraparound(False)  # turn off negative index wrapping for entire func
def chirp_prep(double[:] chirp_in, int len_end, int SHIFT, int SHIFT_POS):
    '''edit chirp'''

    cdef int n
    cdef int d = np.max((chirp_in.shape[0], chirp_in.shape[1]))
    cdef int start_spot = (d - len_end) // 2
    cdef int end_spot = start_spot + len_end
    cdef short[:] chirp_int = np.zeros((len(chirp_in),), dtype=np.int16)
    for n in range(len(chirp_in)):
        chirp_int[n] = np.int16(chirp_in[n] / (2**SHIFT))
    return chirp_int.base[start_spot:end_spot]


@cython.boundscheck(False)  # turn off bounds-checking for entire function
# @cython.wraparound(False)  # turn off negative index wrapping for entire func
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

    cdef short[:, :, :, :] wera = np.empty((IQ, MTC, NANT, NCHIRP), dtype=np.short)

    cdef int i0, i2, k
    cdef int[:] indata
    cdef double[:, :] data
    cdef double[:, :] sdata
    cdef double[:, :] datac
    cdef short[:, :, :] wera1
    cdef short[:, :, :] werac
    cdef double[:] out1
    cdef double[:] out2
    cdef short[:] out3
    cdef short[:] out4

    for ichirp in range(0, NCHIRP):

        # initialize variables per chirp
        sdata = np.empty((NCHAN * IQ, MTL))
        datac = np.empty((NCHAN * IQ, MTCL))
        wera1 = np.empty((IQ, MT, NANT), dtype=np.short)
        werac = np.empty((IQ, MTC, NANT), dtype=np.short)

        #  move back a bit in file to extend chirp so filter works cleanly
        fi.seek(-4 * NCHAN * IQ * SHIFT_POS, 1)
        indata = np.fromfile(fi, np.int32, NCHAN * IQ * (MT * OVER + SHIFT_POS))
        # data = indata.reshape((NCHAN * IQ, MT * OVER + SHIFT_POS)).astype(ct.c_double)
        data = np.reshape(np.float64(indata), (NCHAN * IQ, MT * OVER + SHIFT_POS))
        # print(data.base)

        # manipulate data for each channel
        for ichan in range(0, NANT * IQ):
            # window, decimate, and unwindow
            out1 = chirp_compress(data[ichan, :], OVER)
            # print(out1.base)
            out2 = chirp_compress(sdata[ichan, :], COMP_FAC)
            # print(out2.base)
            for k in range(MTL):
                sdata[ichan, k] = out1.base[k]
            for k in range(MTCL):
                datac[ichan, k] = out2.base[k]
            # sdata[ichan, :] = chirp_compress(data[ichan, :], OVER)
            # datac[ichan, :] = chirp_compress(sdata[ichan, :], COMP_FAC)

            # reorder channels, shift bits, and move to int16 data
            i0 = np.intc(map[ichan, 2])
            i2 = np.intc(map[ichan, 1])
            out3 = chirp_prep(sdata[ichan, :], MT, SHIFT, SHIFT_POS)
            # print(out3.base)
            out4 = chirp_prep(datac[ichan, :], MTC, SHIFT, SHIFT_POS)
            # print(out4.base)
            for k in range(MT):
                wera1[i0, k, i2] = out3.base[k]
            for k in range(MTC):
                werac[i0, k, i2] = out4.base[k]
            # wera1[i0, :, i2] = chirp_prep(sdata[ichan, :], MT, SHIFT, SHIFT_POS)
            # werac[i0, :, i2] = chirp_prep(datac[ichan, :], MTC, SHIFT, SHIFT_POS)
        # print(werac.base.shape)
        # print(wera.base.shape)
        wera.base[..., ichirp] = werac.base  # store compressed data in 'wera'
        fo.write(wera1)  # write out data to RAW bin output file
    return wera.base
