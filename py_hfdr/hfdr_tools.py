#!/usr/bin/env python3
"""

 Module w/ collection of classes and functions to handle and process HFDR data

"""


class configs:
    '''Container for site-specific processing configuration variables'''
    def __init__(self, params_struc):
        self.name = params_struc['Site'][0][:3]
        self.site_fullname = params_struc['Site'][0]

    def set_configs(self, config_file):
        print("This will overwrite defaults for " + self.name + " configs.")

    def set_defaults(self):
        print("Setting config vars in " + self.name + " to defaults.")


def read_raw_compressed():
    '''Reads a compressed raw file (.npz/.mat)'''


def main():
    print("This is a module to be imported only.")


if __name__ == "__main__":
    main()
