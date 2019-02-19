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

 TODO: try CYTHON to make loop and arrays faster.

 Author: Saulo M. Soares
"""

import os
import argparse
# import sys
import time
import numpy as np
# from scipy import signal
from scipy.io import loadmat, savemat
from hfdr_tools import Configs, chirp_prep, chirp_compress


start_time = time.time()

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
parser.add_argument("-c", "--config_file", action="store", default=None,
                    type=str, help="config file for processing variables")
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

if args.config_file is not None:
    if os.path.isfile(args.config_file):
        print("~ Input proc. config file is: {}".format(args.config_file))
    else:
        raise ValueError('Input config file (.txt) not found!')

# maybe change below to delete the existing file or to allow overwrite?
if (os.path.isfile(args.out_filename + '.bnr') or
   os.path.isfile(args.out_filename + '.mat') or
   os.path.isfile(args.out_filename + '.npz')):
    raise ValueError('Output already exists! Aborting')
else:
    outfilename = args.out_filename

# ----------------------------
# Populate variables using the new class

params_in_hfdr = loadmat(params_filein)

if args.config_file is None:
    site_conf = Configs(params_in_hfdr)
else:
    site_conf = Configs(params_in_hfdr, config_file=args.config_file)

NCHAN = site_conf.vars.NCHAN  # number of dtacq A/D channel pairs
NANT = site_conf.vars.NANT  # WERA number of antennas, WHY IS SAME AS ABOVE/IS IT ALWAYS?
MT = site_conf.vars.MT  # number of WERA samples per chirp 1920 (tipical high res) or 3072 (typical long range)

IQ = site_conf.vars.IQ  # number of channels to make a pair
OVER = site_conf.vars.OVER  # dtacq oversampling rate
NCHIRP = site_conf.vars.NCHIRP  # number of chirps
SKIP = site_conf.vars.SKIP  # number of chirps skipped
COMP_FAC = site_conf.vars.COMP_FAC  # extra compression factor
FIRSKIP = site_conf.vars.FIRSKIP  # no of samples skipped before first chirp due to FIR transcient
SHIFT_POS = site_conf.vars.SHIFT_POS  # total no of samples to add to chirp for clean filtering
SHIFT = site_conf.vars.SHIFT  # number of bits lost when converting 32->16 bits
HEADTAG = site_conf.vars.HEADTAG  # part of the header necessary
IQORDER = site_conf.vars.IQORDER  # ordering of I and Q channels; use 'norm' or 'swap'

MTL = site_conf.vars.MTL  # need care that this being int and rounded right
MTCL = site_conf.vars.MTCL
MTC = site_conf.vars.MTC

Map = site_conf.MAP  # call to local function load_map, set map array

# --------------------------------------------------
# Open and prepare input file

fi = open(dta_filein, 'rb')  # fi = open(dta_filein, 'r', 'ieee-le')
# or do this: fi_np = np.fromfile(dta_filein, dtype='<f4', count=-1)

# fseek(fi, 0, 'eof')  # this indicates we go to end of file, with zero offset
fi.seek(0, 2)  # trying to reproduce that

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

# read, manipulate, and write data (CYTHONIZE!)

# wera = loop_chirps(fi, fo, site_conf)
wera = np.empty((IQ, MTC, NANT, NCHIRP), dtype=np.short)

for ichirp in range(0, NCHIRP):

    # initialize variables per chirp
    sdata = np.empty((NCHAN * IQ, MTL))
    datac = np.empty((NCHAN * IQ, MTCL))
    wera1 = np.empty((IQ, MT, NANT), dtype=np.short)
    werac = np.empty((IQ, MTC, NANT), dtype=np.short)

    #  move back a bit in file to extend chirp so filter works cleanly
    fi.seek(-4 * NCHAN * IQ * SHIFT_POS, 1)
    indata = np.fromfile(fi, '<i4', NCHAN * IQ * (MT * OVER + SHIFT_POS))
    data = indata.reshape((NCHAN * IQ, MT * OVER + SHIFT_POS), order='F').astype(np.double)
    # data = np.reshape(np.float64(indata), (NCHAN * IQ, MT * OVER + SHIFT_POS), order='F')

    # manipulate data for each channel
    for ichan in range(0, NANT * IQ):
        # window, decimate, and unwindow
        sdata[ichan] = chirp_compress(data[ichan], OVER)
        datac[ichan] = chirp_compress(sdata[ichan], COMP_FAC)

        # reorder channels, shift bits, and move to int16 data
        i0 = np.int(Map[ichan, 2])
        i2 = np.int(Map[ichan, 1])
        wera1[i0, :, i2] = chirp_prep(sdata[ichan], MT, SHIFT, SHIFT_POS)
        werac[i0, :, i2] = chirp_prep(datac[ichan], MTC, SHIFT, SHIFT_POS)
    wera[..., ichirp] = werac  # store compressed data in 'wera'
    fo.write(wera1)  # write out data to RAW bin output file

fo.close()
fi.close()

# ----------------------------------------------------
# Write the output mat (npz) file

np.savez_compressed(outfilename + '.npz', wera=wera, proc_configs=site_conf)
savemat(outfilename + '.mat', {'wera': wera, 'proc_configs': site_conf})
print("--- %s seconds ---" % (time.time() - start_time))
