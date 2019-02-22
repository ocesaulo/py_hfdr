#!/usr/bin/env python3
'''
   Packgage to do spectra and plot for HFDR data
   This is a module tha can be imported or excuted.
'''


import os
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from hfdr_tools import read_raw_compressed, spectra_raw_compressed


def main(arg):
    '''Either imported or excuted, read raw compressed file passed as arg,
    calculate spectra and do standarized plots.
    '''
    # print("Template figures, do the plots and save figures of raw radar data.")
    print("Plots for raw file: " + arg)

    cmin, cmax = -160., 0

    rawdata, confs = read_raw_compressed(arg)
    Frq, Rng, DSA, PHA = spectra_raw_compressed(rawdata, confs)
    filename = os.path.basename(arg)

    Rang_lim = confs.vars.Nranges * confs.vars.res / 1e3
    Nant = confs.vars.NANT
    Fr = confs.vars.Fr  # TransmitCenterFreqMHz

    date_stamp = datetime.strptime(filename[:11], '%Y%j%H%M')
    info = filename[:11] + ' ' + filename[12:15] + ', Fr: ' + str(Fr)

    fig, AX, fs = set_fig_layout(Nant)
    col_bar_ax = fig.add_axes([.925, .075, .015, .8])

    for n, ax in enumerate(AX.flatten()):
        cp = ax.pcolormesh(Frq, Rng, 20 * np.log10(DSA[n]), vmin=cmin, vmax=cmax)
        ax.set_title('Doppler-range spectrum for antenna ' + str(n + 1), fontsize=fs)
        ax.set_ylim([0, Rang_lim])
    [a.set_xlabel('Doppler frequency [Hz]') for a in AX[-1, :]]
    [a.set_ylabel('Range [km]') for a in AX[:, 0]]
    clb = plt.colorbar(cp, cax=col_bar_ax)
    clb.ax.set_title('[db]')
    plt.annotate(s=info, xy=(.75, .025), xycoords='figure fraction')
    plt.annotate(s=date_stamp, xy=(.08, .025), xycoords='figure fraction')
    plt.savefig('../data/test_fig.png', bbox_inches='tight')


def set_fig_layout(Nant, width=11., height=8.5):
    '''Set layout of output figures using defaults, return handles for mods'''
    fig_size = (width, height)
    if Nant == 1:
        r, c = 1, 1
        fs = 'medium'
    elif Nant == 2:
        r, c = 2, 1
        fs = 'medium'
    elif Nant == 3 or Nant == 4:
        r, c = 2, 2
        fs = 'small'
    elif Nant == 5 or Nant == 6:
        r, c = 2, 3
        fs = 'smaller'
    elif Nant == 7 or Nant == 8:
        r, c = 2, 4
        fs = 'x-small'
    fig, AX = plt.subplots(r, c, sharex=True, sharey=True, squeeze=True,
                           figsize=fig_size, facecolor='w')
    return fig, AX, fs


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
