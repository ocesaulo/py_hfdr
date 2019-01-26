#!/usr/bin/env python3
"""

 Module w/ collection of classes and functions to handle and process HFDR data

"""


import numpy as np


class Configs:
    '''Container for site-specific processing configuration variables'''
    def __init__(self, params_struc, config_file=None):
        self.name = params_struc['Site'][0][:3]
        self.site_fullname = params_struc['Site'][0]

        self.vars = self.Vars()
        self.vars.NANT = np.int(params_struc['Ant'])
        # standard_keys = ['IQ', 'OVER', 'SKIP', 'NCHIRP', 'FIRSKIP', 'SHIFT',
        #                  'SHIFT_POS', 'COMP_FAC', 'HEADTAG', 'IQORDER', 'MT',
        #                  'NCHAN', 'NANT', 'MTC', 'MTL', 'MTCL']
        # self.vars = {key:None for key in standard_keys}
        self.vars.NCHAN = np.int(params_struc['Ant'])
        self.vars.MT = np.int(params_struc['MT'])

        if config_file:
            self.set_configs(config_file)
        else:
            self.set_defaults()

        self.vars.MTL = self.vars.MT + self.vars.SHIFT_POS // self.vars.OVER
        self.vars.MTCL = np.int(np.ceil(self.vars.MTL / self.vars.COMP_FAC))
        self.vars.MTC = np.int(np.ceil(self.vars.MT / self.vars.COMP_FAC))
        # self.map = load_map(IQORDER)  # call to local function load_map, set map array

    class Vars:
        pass

    def set_configs(self, config_file):
        print("This will overwrite defaults for " + self.name + " configs.")

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


def read_raw_compressed():
    '''Reads a compressed raw file (.npz/.mat)'''


def main():
    print("This is a module to be imported only.")


if __name__ == "__main__":
    main()
