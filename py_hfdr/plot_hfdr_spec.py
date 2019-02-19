#!/usr/bin/env python3
'''
   Packgage to do spectra and plot for HFDR data
   This is a module tha can be imported or excuted.
'''


import numpy as np
from hfdr_tools import read_raw_compressed


def main(arg):
    '''Either imported or excuted, read raw compressed file passed as arg and
    do standard plots.
    '''
    print("Template figures, do the plots and save figures of raw radar data.")
    print("plots for file: " + arg)
    rawdata, configs = read_raw_compressed(arg)
    nant = configs.vars.NANT


def set_default_fig_layout():
    '''Set layout of output figures using defaults, return handles for mods'''


if __name__ == "__main__":
    import sys
    main(sys.argv)
