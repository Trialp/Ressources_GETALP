#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys
import numpy as np

help_msg = """
Read a score file in numpy format.
Usage: {0} FILENAME
"""

if __name__ == '__main__':
    try:
        filename = sys.argv[1]
    except (ValueError, IndexError):
        sys.exit(help_msg.strip('\n').format(sys.argv[0]))

    array = np.load(open(filename))
    print '\n'.join(map(str, array['bleu_scores']))