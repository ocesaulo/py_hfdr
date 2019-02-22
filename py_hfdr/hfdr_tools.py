#!/usr/bin/env python3
"""

 Module w/ collection of classes and functions to handle and process HFDR data

"""


import os
import numpy as np
from scipy.signal import decimate
from scipy.signal.windows import blackmanharris
from scipy.io import loadmat
from scipy.fftpack import fft, fftshift, fftfreq


class Configs:
    '''Container for site-specific processing configuration variables'''
    def __init__(self, params_struc, config_file=None):
        self.name = params_struc['Site'][:3]
        self.site_fullname = params_struc['Site']
        self.manufacturer = params_struc['Manufacturer']
        self.origin = params_struc['Origin']
        self.timezone = params_struc['TimeZone']

        self.vars = self.Vars()
        self.vars.NANT = np.int(params_struc['Ant'])
        # standard_keys = ['IQ', 'OVER', 'SKIP', 'NCHIRP', 'FIRSKIP', 'SHIFT',
        #                  'SHIFT_POS', 'COMP_FAC', 'HEADTAG', 'IQORDER', 'MT',
        #                  'NCHAN', 'NANT', 'MTC', 'MTL', 'MTCL']
        # self.vars = {key:None for key in standard_keys}
        self.vars.NCHAN = np.int(params_struc['Ant'])
        self.vars.MT = np.int(params_struc['MT'])

        self.vars.Fr = np.float64(params_struc['TransmitCenterFreqMHz'])
        # duration of one chirp [sec]:
        self.vars.T1 = 1 / np.float64(params_struc['TransmitSweepRateHz'])
        # bandwidth [KHz]:
        self.vars.hf_bw = np.float64(params_struc['TransmitBandwidthKHz'])

        # self.vars.res = c / (2 * hf_bw)  # resolution of range cell [m]
        self.vars.res = np.float64(params_struc['RangeResolutionKMeters'])*1e3
        self.vars.Nord = np.float64(params_struc['TrueNorth'])
        self.vars.Nranges = np.float64(params_struc['Nranges'])

        if config_file:
            self.set_configs(config_file)
        else:
            self.set_defaults()

        self.vars.MTL = self.vars.MT + self.vars.SHIFT_POS // self.vars.OVER
        self.vars.MTCL = np.int(np.ceil(self.vars.MTL / self.vars.COMP_FAC))
        self.vars.MTC = np.int(np.ceil(self.vars.MT / self.vars.COMP_FAC))
        self.MAP = load_map(self.vars.IQORDER)  # call to local func load_map, set map array

    class Vars:
        pass

    def set_configs(self, config_file):
        print("This will overwrite defaults for " + self.name + " configs.")
        print("Not coded yet...")

    def set_defaults(self):
        print("Setting config vars in " + self.name + " to defaults.")
        self.vars.IQ = 2
        self.vars.OVER = 2
        self.vars.SKIP = 1
        self.vars.NCHIRP = 2048
        self.vars.FIRSKIP = 28
        self.vars.COMP_FAC = 8
        self.vars.SHIFT = 18
        self.vars.SHIFT_POS = self.vars.COMP_FAC * 20 * 2
        self.vars.HEADTAG = '2048 SAMPLES   '
        self.vars.IQORDER = 'radcelf'


class Phys_Constants:
    ''' Container for physics constants of relevance to HFDR '''
    g = 9.81  # gravitational acceleration
    c0 = 2.99792458e8  # speed of light in vacuum
    c = c0 * (1 - 300 / 1e6)  # speed of radio waves in moist tropical air


def load_map(IQORDER):
    ''' Make map for channel remapping '''

    if IQORDER == 'norm':
        IQCHAN = np.r_[np.ones((8, 1)), 2 * np.ones((8, 1)), np.ones((8, 1)),
                       2 * np.ones((8, 1))]
    elif IQORDER == 'swap':
        IQCHAN = np.r_[2 * np.ones((8, 1)), np.ones((8, 1)),
                       2 * np.ones((8, 1)), np.ones((8, 1))]
    elif IQORDER == 'radcelf':
        IQCHAN = np.reshape(np.r_[np.ones((1, 8)), 2 * np.ones((1, 8))],
                            8 * 2, 1)
        DDS_OUT = np.array([np.arange(1, 9), np.arange(1, 9)])
    else:
        raise ValueError('Incorrect choice for IQORDER')

    map = np.zeros((8 * 2, 3), dtype=np.int)
    map[:, 0] = np.r_[1:8 * 2 + 1]
    map[:, 1] = DDS_OUT.T.ravel()  # will break, as var only exists in one cond
    map[:, 2] = IQCHAN
    return map - 1


def chirp_compress(chirp_in, compression_factor):
    '''window, decimate and unapply window to chirp'''

    d = np.max(chirp_in.shape)
    w = blackmanharris(d).T
    w1 = blackmanharris(np.int(np.ceil(d / compression_factor))).T
    wc = chirp_in * w
    dc = decimate(wc, compression_factor)
    return dc / w1


def chirp_prep(chirp_in, len_end, SHIFT, SHIFT_POS):
    '''clip chirp array and set it to int16'''

    d = np.max(chirp_in.shape)
    # start_spot = (d - len_end) // 2 + 1  # indexing is likely off (matlab)
    start_spot = (d - len_end) // 2
    # end_spot = start_spot + len_end - 1  # indexing is likely off
    end_spot = start_spot + len_end
    chirp_out = chirp_in / (2**SHIFT)
    return chirp_out[start_spot:end_spot].astype(np.int16)


def read_raw_compressed(filename):
    '''Reads a compressed raw file (.npz/.mat), outputs double precision'''
    if os.path.isfile(filename):
        pass
    if filename[-3:] == 'npz':
        rawdata = np.load(filename)
        proc_configs = rawdata['proc_configs'][0]
    elif filename[-3:] == 'mat':
        rawdata = loadmat(filename, squeeze_me=True)
        proc_configs = rawdata['proc_configs'][0][0]
    else:
        raise ValueError('Incorrect compressed raw file type (given as str)')
    wera = rawdata['wera'].astype(np.float64)
    confs = proc_configs  # need to structure this in mat
    return wera, confs


def spectra_raw_compressed(rawdata, configs):
    '''Calculate spectra of compressed output of HFDR into spectra '''

    Nant = configs.vars.NANT
    N1 = configs.vars.MTC  # number of samples per chirp - yaxis
    dt1 = configs.vars.T1 / N1  # sampling time [sec] - yaxis
    df1 = 1 / (N1 * dt1)  # fundamental frequency [Hz] - yaxis
    N2 = configs.vars.NCHIRP  # number of samples (total numbers of chirps)
    dt2 = configs.vars.T1  # sampling time [sec] (chirp duration) - xaxis
    df2 = 1 / (N2*dt2)  # fundamental frequency [Hz] - xaxis
    res = configs.vars.res

    w1 = blackmanharris(N1)  # 1st window
    w2 = blackmanharris(N2)  # 2nd window for doppler
    gain = np.empty((Nant,))
    DSA = np.empty((Nant, N1 // 2 + 1, N2))
    PHAF = np.empty((Nant, N1 // 2 + 1, N2))

    for n in range(0, Nant):
        IC = np.squeeze(rawdata[0, :, n, :])  # channel I [sample x chirp]
        QC = np.squeeze(rawdata[1, :, n, :])  # channel Q [sample x chirp]
        IQ = IC + 1j * QC

        # range-resolving Fourier transform
        IQw = IQ * w1[:, np.newaxis]  # Windowing
        F1 = fft(IQw, N1, axis=0) / N1  # first FFT of samples per chirp

        # Reorder fourier coefficients (fold back neg. part of the spectrum)
        F1r = fftshift(F1, 0)

        # Power spectrum of the first fourier transform
        PS1 = np.abs(F1r)**2

        # Antenna gain
        gain[n] = np.sqrt(PS1.mean())

        # Make Doppler-resolving Fourier transform
        F1rw = F1r * w2[np.newaxis, :]

        F2 = fft(F1rw, N2, axis=-1) / N2  # 2nd FOURIER TRANSFORM

        # Reorder fourier coefficients (fold back negat part of the spectrum)
        # F2r = 2 * F2[:, :N2//2 + 1]  # dont think that's what they are doing
        F2r = fftshift(F2, -1)

        # Power spectrum of the second fourier transform
        PS2 = np.abs(F2r)**2

        # Take only the neg part of the doppler spectrum (SORT) and rotate it. Also ignoring RFI
        DPS = np.rot90(PS2[:np.int(np.floor(N1/2)) + 1, :], 2)

        PHA = np.angle(F2r, deg=True)
        PH = np.rot90(PHA[:np.int(np.floor(N1/2)) + 1, :], 2)

        # store:
        PHAF[n, :, :] = PH
        DSA[n, :, :] = DPS

    # freq/range arrays used when plotting Range-Doppler spectrum
    fvec = fftshift(fftfreq(N2, dt2))  # doppler frequency vector
    dr = np.round(res / 1e3, 1)
    rvec = np.r_[:np.floor(N1 / 2)+1] * dr  # pos range vector [km]
    fvec = np.concatenate((fvec[np.newaxis] - df2/2,
                           fvec[-1]*np.ones((1, 1)) + df2/2), axis=1)
    rvec = np.concatenate((rvec[:, np.newaxis] - dr/2,
                           rvec[-1]*np.ones((1, 1)) + dr/2), axis=0)
    Frq, Rng = np.meshgrid(fvec, rvec)  # only needed for plots
    return Frq, Rng, DSA, PHAF


def main():
    print("This is a module to be imported only.")


if __name__ == "__main__":
    main()
