#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
from collections import Counter
import sys

if __name__ == '__main__':
    line_count = 0
    word_count = 0
    voc = Counter()

    for line in sys.stdin:
        line_count += 1
        for word in line.split():
            voc[word] += 1
            word_count += 1

    print '{0} lines, {1} tokens, {2} unique tokens'.format(line_count, word_count, len(voc))
    print 'Average line length: {:.2f}'.format(word_count / line_count)

    min_occ = 5
    if min_occ > 1:
        rare = sum(1 for v in voc.itervalues() if v < min_occ)
        print '{0} tokens have less than {1} occurrences'.format(rare, min_occ)
    
    ultra_rare = sum(1 for v in voc.itervalues() if v == 1)
    print '{0} tokens appear exactly once'.format(ultra_rare)

    max_size = 30000
    coverage = sum(sorted(voc.values(), reverse=True)[:max_size]) / word_count
    print 'The top {0} tokens account for {1:.1f}% of the corpus'.format(max_size, coverage * 100)


