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
        self.MAP = load_map(self.vars.IQORDER)  # call to local func load_map, set map array

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


def read_raw_compressed():
    '''Reads a compressed raw file (.npz/.mat)'''


def main():
    print("This is a module to be imported only.")


if __name__ == "__main__":
    main()
