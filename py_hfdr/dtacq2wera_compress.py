"""
 Converts dta file to RAW file and compressed mat file; Originally coded as
 function

 change parameters in beginning to suit site; TODO: parameter input separate

 Formally ran as:
 $matlab -nodisplay -r "addpath('/pathtofunction');dtacq2wera('$infile','$rawfile','$outfile','$timefile','$headfile');quit;"

 TODO: This will probably turn into a script (bin); input will be a raw file,
 a params file, a header file and others as needed

 Adapted to radcelf

"""

import os
import sys
import numpy as np
from scipy.io import loadmat, savemat


# ----------------------------------------------
# local functions (may rethink)

def load_map(IQORDER):
    ''' Make map for channel remapping '''
    # global IQORDER
    # switch IQORDER
    if IQORDER == 'norm':
        IQCHAN=[ones(8,1); 2*ones(8,1); ones(8,1); 2*ones(8,1)];
    elif IQORDE == 'swap':
        IQCHAN=[2*ones(8,1); ones(8,1); 2*ones(8,1); ones(8,1)];
    elif IQORDER == 'radcelf':
        IQCHAN=reshape([ones(1,8); 2*ones(1,8)],8*2,1);
        DDS_OUT=  [(1:8).T (1:8).T]
    else:
        raise ValueError('Incorrect choice for IQORDER')

    MAP = zeros(8*2,3)
    MAP[:, 0] = [1:8*2]
    MAP[:, 1] = DDS_OUT # not sure, as var only exists in one cond
    MAP[:, 2] = IQCHAN
    return MAP


def chirp_compress(chirp_in, compression_factor):
    '''window, decimate and unapply window to chirp'''
    d = np.max(size(chirp_in))
    w = scipy.signal.windows.blackmanharris(d).T
    w1 = scipy.windows.blackmanharris(np.ceil(d / compression_factor)).T
    wc = chirp_in.*w
    dc = scipy.signal.decimate(wc, compression_factor)
    return dc./w1


def chirp_prep(chirp_in, len_end, SHIFT, SHIFT_POS):
    '''global SHIFT SHIFT_POS'''
    d = length(chirp_in)
    start_spot = (d - len_end) / 2 + 1
    end_spot = start_spot + len_end - 1
    chirp_int = chirp_in / (2**SHIFT)
    return np.int16(chirp_int[start_spot:end_spot])


# ----------------------------
# Read in params file (this input needs change, args to script or env vars)
# TOD_O need error (in arg) handling for the input files

path_to_hfdr = '/Users/saulo/Work/Projects/Software/hfdr_tools'

dta_in_file = 'test_data/20183412100_jbo.dta'
dta_filein = os.path.join(path_to_hfdr, dta_in_file)

params_hfdr_infile = 'params_jbo.mat'
params_path_infile = os.path.join(path_to_hfdr, params_hfdr_infile)

params_in_hfdr = loadmat(params_path_infile)

time_infile = 'test_data/20183412100_jbo.hdr'  # unsure if this the right file
time_path_infile = os.path.join(path_to_hfdr, time_infile)

header_infile = 'test_data/jbo.hdr'  # unsure if this the right file
header_path_infile = os.path.join(path_to_hfdr, header_infile)

rawout_filename = '../data/sometest.bnr'  # unsure of the output format

# Populate variables (some hard coded? why?)
# again the extraction of vars from mat file will need to change
# as there must be a more elegant way

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
FIRSKIP = 28  # number of samples skipped before first chirp due to FIR transcient
SHIFT_POS = COMP_FAC * 20 * 2  # total number of samples to add to chirp to make clean filtering
# %SHIFT = 16  # number of bits lost when converting 32->16 bits
SHIFT = 18
HEADTAG = '2048 SAMPLES   '  # part of the header necessary
IQORDER = 'radcelf'  # ordering of I and Q channels; use 'norm' or 'swap'

MTL = MT + SHIFT_POS / OVER

MTCL = np.ceil(MTL / COMP_FAC)
MTC = np.ceil(MT / COMP_FAC)
# map = load_map;  # this needs clarification; its a function local

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
# clear ft fh

# write header to output file
fo = open(rawout_filename, 'wb')  # supposed to be bin or text?? I think bin
# fo.write((HEADTAG + timed + ' header'), '*char')  # ??? is header used or no
fo.write((HEADTAG + timed + header))  # not sure I'm doing the right
# clear header timed filein filetime filehead rawout
