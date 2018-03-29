#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
from itertools import chain, count, islice
from collections import Counter
from operator import itemgetter
import sys
import cPickle

help_msg = """
    Usage: {0} OUT_FILE MAX_SIZE MIN_COUNT < TEXT_FILE
"""
if __name__ == '__main__':
    try:
        out_file = sys.argv[1]
        max_size, min_count = map(int, sys.argv[2:])
        if max_size <= 0:
            max_size = sys.maxint
    except IndexError:
        sys.exit(help_msg.strip('\n').format(sys.argv[0]))

    counter = Counter(chain.from_iterable(line.split() for line in sys.stdin))
    words = map(itemgetter(0), sorted(((k, v) for k, v in counter.iteritems() if v >= min_count), key=itemgetter(1),
        reverse=True))[:max_size]
    voc = dict(zip(words, count()))

    with open(out_file, 'w') as f:
        cPickle.dump(voc, f)

