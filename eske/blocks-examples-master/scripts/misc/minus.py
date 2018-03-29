#!/bin/env python2

import sys

if __name__ == '__main__':
    help_msg = """
    Usage: {0} FILE1 FILE2
    Outputs all lines in FILE1 that do not belong to FILE2.
    """
    try:
        filename1, filename2 = sys.argv[1:]
    except IndexError:
        sys.exit(help_msg.strip('\n').format(sys.argv[0]))

    lines = set(open(filename2))
    for line in open(filename1):
        if line not in lines:
            print line,

