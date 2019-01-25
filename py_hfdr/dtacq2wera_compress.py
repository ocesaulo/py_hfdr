"""
 Converts dta file to RAW file and compressed mat file; Originally coded as
 function

 change parameters in beginning to suit site; TODO: parameter input separate

 Formally ran as:
 $matlab -nodisplay -r "addpath('/
 pathtofunction');dtacq2wera('$infile','$rawfile','$outfile','$timefile',/
 '$headfile');quit;"

 TODO: This will probably turn into a script (bin); input will be a raw file,
 a params file, a header file and others as needed

 Adapted to radcelf

"""

import os
import sys
import numpy as np
from scipy import signal
from scipy.io import loadmat, savemat


# ----------------------------------------------
# local functions (may rethink)

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

    map = np.zeros((8 * 2, 3))
    map[:, 0] = np.r_[1:8 * 2 + 1]
    map[:, 1] = DDS_OUT.T.ravel()  # will break, as var only exists in one cond
    map[:, 2] = IQCHAN
    return map


def chirp_compress(chirp_in, compression_factor):
    '''window, decimate and unapply window to chirp'''

    d = np.max(chirp_in.shape)
    w = signal.windows.blackmanharris(d).T
    w1 = signal.windows.blackmanharris(np.ceil(d / compression_factor)).T
    wc = chirp_in * w
    dc = signal.decimate(wc, compression_factor)
    return dc / w1


def chirp_prep(chirp_in, len_end, SHIFT, SHIFT_POS):
    '''global SHIFT SHIFT_POS'''

    d = np.max(chirp_in.shape)
    start_spot = (d - len_end) // 2 + 1
    end_spot = start_spot + len_end - 1
    chirp_int = chirp_in // (2**SHIFT)
    return np.int16(chirp_int[start_spot:end_spot])


# ----------------------------
# Read in params file (this input needs change, args to script or env vars)
# TOD_O need error (in arg) handling for the input files

path_to_hfdr = '/Users/saulo/Work/Projects/Software/hfdr_tools'

dta_in_file = 'test_data/20183412100_jbo.dta'
dta_filein = os.path.join(path_to_hfdr, dta_in_file)

params_hfdr_infile = 'params_jbo.mat'
params_path_infile = os.path.join(path_to_hfdr, params_hfdr_infile)

time_infile = 'test_data/20183412100_jbo.hdr'  # unsure if this the right file
time_path_infile = os.path.join(path_to_hfdr, time_infile)

header_infile = 'test_data/jbo.hdr'  # unsure if this the right file
header_path_infile = os.path.join(path_to_hfdr, header_infile)

rawout_filename = '../data/sometest.bnr'  # unsure of the output format

# Populate variables (some hard coded? why? -> this will need to change)
# again the extraction of vars from mat file will need to change
# as there must be a more elegant way

params_in_hfdr = loadmat(params_path_infile)

NCHAN = np.int(params_in_hfdr['Ant'])  # number of dtacq A/D channel pairs
NANT = np.int(params_in_hfdr['Ant'])  # WERA number of antennas (WHY IS SAME
#                                        AS ABOVE/IS IT ALWAYS?
MT = np.int(params_in_hfdr['MT'])  # number of WERA samples per chirp 1920
#                                     (tipical high res) or 3072 (typical
#                                      long range)

IQ = 2  # number of channels to make a pair
OVER = 2  # dtacq oversampling rate
NCHIRP = 2048  # number of chirps
SKIP = 1  # number of chirps skipped
COMP_FAC = 8  # extra compression factor
FIRSKIP = 28  # no of samples skipped before first chirp due to FIR transcient
SHIFT_POS = COMP_FAC * 20 * 2  # total no of samples to add to chirp for clean filtering
# %SHIFT = 16  # number of bits lost when converting 32->16 bits
SHIFT = 18
HEADTAG = '2048 SAMPLES   '  # part of the header necessary
IQORDER = 'radcelf'  # ordering of I and Q channels; use 'norm' or 'swap'

MTL = MT + SHIFT_POS / OVER

MTCL = np.int(np.ceil(MTL / COMP_FAC))
MTC = np.int(np.ceil(MT / COMP_FAC))

map = load_map(IQORDER)  # call to local function load_map, set map array

# --------------------------------------------------
# Open and prepare input file
# again needs to revise

# fi = open(dta_filein, 'r', 'ieee-le')
fi = open(dta_filein, 'rb')
# or do this:
# fi_np = np.fromfile(dta_filein, dtype='<f4', count=-1)

# fseek(fi, 0, 'eof')  # this indicates we go to end of file, with zero offset
fi.seek(0, 2)  # trying to reproduce this

# Check the size of the file:
pos = fi.tell()
size_crit = 4 * ((NCHAN * IQ * OVER * MT * (NCHIRP + SKIP)) +
                 (FIRSKIP * NCHAN * IQ))

if pos < size_crit:
    raise ValueError('File size incorrect: Input file too small')

# Move to after FIR transcient and first chirp:
new_pos = 4 * (((SHIFT_POS // 2 + FIRSKIP) * NCHAN * IQ) +
               (SKIP * NCHAN * IQ * OVER * MT))
fi.seek(new_pos, 0)

# ----------------------------------------------------
# Prepare header and output file:
# get time of acquisition for header

ft = open(time_path_infile, 'r')  # may not be a binary file: switched to r
timed = ft.read()
ft.close()

# get WERA header

fh = open(header_path_infile, 'r')  # does not seem to be binary file?!
header = fh.read()
fh.close()

# write header to output file

fo = open(rawout_filename, 'wb')  # supposed to be bin or text?? I think bin
fo.write((HEADTAG + timed + header).encode('ascii'))  # not sure it will work

# read, manipulate, and write data

wera = np.zeros((IQ, MTC, NANT, NCHIRP), dtype=np.int16)

for ichirp in range(1, NCHIRP):

    #  move back a bit in file to extend chirp so filter works cleanly
    fi.seek(-4 * NCHAN * IQ * SHIFT_POS, 1)
    data = np.fromfile(fi, np.int32, NCHAN * IQ * (MT * OVER + SHIFT_POS))
    data = data.reshape((NCHAN * IQ, MT * OVER + SHIFT_POS)).astype(np.float64)

    # initialize variables per chirp
    sdata = np.float64(np.zeros(NCHAN * IQ, MTL))
    datac = np.float64(np.zeros(NCHAN * IQ, MTCL))
    wera1 = np.int16(np.zeros(IQ, MT, NANT))
    werac = np.int16(np.zeros(IQ, MTC, NANT))

    # manipulate data for each channel
    for ichan in range(1, NANT * IQ):
        # window, decimate, and unwindow
        sdata[ichan, :] = chirp_compress(data[ichan, :], OVER)
        datac[ichan, :] = chirp_compress(sdata[ichan, :], COMP_FAC)
        # reorder channels, shift bits, and move to int16 data
        wera1[map[ichan, 3], :, map[ichan, 2]] = chirp_prep(sdata[ichan, :], MT)
        werac[map[ichan, 3], :, map[ichan, 2]] = chirp_prep(datac[ichan, :], MTC)
    wera[..., ichirp] = werac  # store compressed data in 'wera'
    fo.write(wera1)  # write out data to RAW output file
fo.close()
fi.close()

# ----------------------------------------------------
# Write the output mat (npz) file

savemat(rawout_filename[:-3] + 'mat', wera=wera)
np.savez_compressed(rawout_filename[:-3] + 'npz', wera=wera)
