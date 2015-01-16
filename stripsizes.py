from __future__ import print_function
import re
import sys

CONFIG = 'run.conf'

if __name__ == '__main__':
    f = open(CONFIG)

    for l in f:
        if sys.argv[1] == 'NX':
            if 'NX' in l:
                print(l.split()[2][:-1])

        if sys.argv[1] == 'NY':
            if 'NY' in l:
                print(l.split()[2][:-1])

        if sys.argv[1] == 'NZ':
            if 'NZ' in l:
                print(l.split()[2][:-1])
