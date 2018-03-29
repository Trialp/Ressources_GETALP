#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
from itertools import chain
from collections import Counter
import sys
import cPickle

help_msg = """
    Usage: {0} VOC_FILE < TEXT_FILE
"""

if __name__ == '__main__':
    try:
        voc_file = sys.argv[1]
    except IndexError:
        sys.exit(help_msg.strip('\n').format(sys.argv[0]))
    
    with open(voc_file) as f:
        voc = cPickle.load(f)
    counter = Counter(chain.from_iterable(line.split() for line in sys.stdin))

    word_count = sum(counter.itervalues()) 
    covered = [v for k, v in counter.iteritems() if k in voc]
    print '{0}/{1} vocabulary coverage, {2:.2f}% corpus coverage'.format(len(covered), len(counter), 100 * sum(covered) / word_count)

