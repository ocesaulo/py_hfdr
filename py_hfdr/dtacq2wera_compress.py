#!/usr/bin/env python3
"""
 Converts dta file to RAW file and compressed mat and npz files. Change inputs
 and variables to adapt to radar site configurations.

 usage: $python dtacq2wera_compress ./file.dta ./params.mat ./201009090000_site.hdr
         ./site.hdr output_filename

 TODO: parameter input must be separate, need to avoid hard coded radar site
 config. vars: USE A CLASS and import a module that contains it; class to be
 partially filled from the params file.
 Currently only radcelf option works (bug): ask Xavier why

 TODO: This will probably turn into an excutable script (bin); input will be a
 raw .dta file, a params file, a header file and others as needed. Output will
 be 3 files: a raw binary .bnr, a matlab file .mat and numpy file .npz.

 TODO: Improve error handling in argument passing and checks. Currently if the
 output files exist abort.

 Author: Saulo M. Soares
"""

import os
import argparse
# import sys
import numpy as np
from scipy import signal
from scipy.io import loadmat, savemat


# ----------------------------------------------
# local functions (may rethink and go into a module to be imported)

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
    w = signal.windows.blackmanharris(d).T
    w1 = signal.windows.blackmanharris(
                                       np.int(np.ceil(d / compression_factor))
                                       ).T
    wc = chirp_in * w
    dc = signal.decimate(wc, compression_factor)
    return dc / w1


def chirp_prep(chirp_in, len_end, SHIFT, SHIFT_POS):
    '''size it right? not sure what this is doing'''

    d = np.max(chirp_in.shape)
    # start_spot = (d - len_end) // 2 + 1  # indexing is likely off (matlab)
    start_spot = (d - len_end) // 2
    # end_spot = start_spot + len_end - 1  # indexing is likely off
    end_spot = start_spot + len_end
    chirp_int = chirp_in // (2**SHIFT)
    return np.int16(chirp_int[start_spot:end_spot])


# ----------------------------
# Read in input files

parser = argparse.ArgumentParser(description=("Converts .dta file to " +
                                              ".bnr, .mat and .npz files."),
                                 epilog=("usage: dtacq2wera_compress " +
                                         "./file.dta ./params.mat " +
                                         "./YYYYMMDDHHMM_sitename.hdr " +
                                         "./sitename.hdr " +
                                         "YYYYMMDDHHMM_sitename")
                                 )
parser.add_argument("dta_file", type=str, help="input .dta file w/ path")
parser.add_argument("params_file", type=str,
                    help="input params file (.mat) w/ full path")
parser.add_argument("time_file", type=str,
                    help="input time file (TIMESTAMP_sitename.hdr) w/ path")
parser.add_argument("header_file", type=str,
                    help="input header file (sitename.hdr) w/ path")
parser.add_argument("out_filename", type=str,
                    help="output file name w/o ext (.bnr .mat and .npz)")
args = parser.parse_args()

# do the checks that files exist

if os.path.isfile(args.dta_file):
    print("~ Input file is: {}".format(args.dta_file))
    dta_filein = args.dta_file
else:
    raise ValueError('Input .dta file not found!')

if os.path.isfile(args.params_file):
    print("~ Parameter file is: {}".format(args.params_file))
    params_filein = args.params_file
else:
    raise ValueError('Input params file not found!')

if os.path.isfile(args.time_file):
    print("~ Input time file is: {}".format(args.time_file))
    time_filein = args.time_file
else:
    raise ValueError('Input time file (.hdr) not found!')

if os.path.isfile(args.header_file):
    print("~ Input header file is: {}".format(args.header_file))
    header_filein = args.header_file
else:
    raise ValueError('Input header file (.hdr) not found!')

# maybe change below to delete the existing file or to allow overwrite?
if (os.path.isfile(args.out_filename + '.bnr') or
   os.path.isfile(args.out_filename + '.mat') or
   os.path.isfile(args.out_filename + '.npz')):
    raise ValueError('Output already exists! Aborting')
else:
    outfilename = args.out_filename

# ----------------------------
# Populate variables (some hard coded? why? -> this will need to change)
# again the extraction of vars from mat file will need to change
# as there must be a more elegant way (class that is populated)

params_in_hfdr = loadmat(params_filein)

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

MTL = MT + SHIFT_POS // OVER  # need care that this being int and rounded right

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

ft = open(time_filein, 'r')  # may not be a binary file: switched to r
timed = ft.read()
ft.close()

# get WERA header

fh = open(header_filein, 'r')  # does not seem to be binary file?!
header = fh.read()
fh.close()

# write header to output file

fo = open(outfilename + '.bnr', 'wb')  # supposed to be bin or txt? I think bin
fo.write((HEADTAG + timed + header).encode('ascii'))  # not sure it will work

# read, manipulate, and write data

wera = np.zeros((IQ, MTC, NANT, NCHIRP), dtype=np.int16)

for ichirp in range(0, NCHIRP):

    #  move back a bit in file to extend chirp so filter works cleanly
    fi.seek(-4 * NCHAN * IQ * SHIFT_POS, 1)
    data = np.fromfile(fi, np.int32, NCHAN * IQ * (MT * OVER + SHIFT_POS))
    data = data.reshape((NCHAN * IQ, MT * OVER + SHIFT_POS)).astype(np.float64)

    # initialize variables per chirp
    sdata = np.float64(np.zeros((NCHAN * IQ, MTL)))
    datac = np.float64(np.zeros((NCHAN * IQ, MTCL)))
    wera1 = np.int16(np.zeros((IQ, MT, NANT)))
    werac = np.int16(np.zeros((IQ, MTC, NANT)))

    # manipulate data for each channel
    for ichan in range(0, NANT * IQ):
        # window, decimate, and unwindow
        sdata[ichan, :] = chirp_compress(data[ichan, :], OVER)
        datac[ichan, :] = chirp_compress(sdata[ichan, :], COMP_FAC)
        # reorder channels, shift bits, and move to int16 data
        wera1[map[ichan, 2], :, map[ichan, 1]] = chirp_prep(sdata[ichan, :],
                                                            MT, SHIFT, SHIFT_POS)
        werac[map[ichan, 2], :, map[ichan, 1]] = chirp_prep(datac[ichan, :],
                                                            MTC, SHIFT, SHIFT_POS)
    wera[..., ichirp] = werac  # store compressed data in 'wera'
    fo.write(wera1)  # write out data to RAW bin output file
fo.close()
fi.close()

# ----------------------------------------------------
# Write the output mat (npz) file

savemat(outfilename + '.mat', wera=wera)
np.savez_compressed(outfilename + '.npz', wera=wera)
