#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
import sys
import argparse
import cPickle


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filters out </S> and <S> tokens from a given file.')
    parser.add_argument('infile', nargs='?', type=argparse.FileType('r'), default=sys.stdin,
                        help='Input file. Defaults to standard input.')
    parser.add_argument('--voc', type=argparse.FileType('r'),
                        help='Filter out OOV words according to this dict (cPickle format).')
    parser.add_argument('--no-unk', help='Remove <UNK> tokens.', action='store_true')
    args = parser.parse_args()
  
    voc = args.voc
    if voc is not None:
        voc = cPickle.load(voc)

    filter_out = '</S>', '<S>'
    if args.no_unk:
        filter_out += '<UNK>',

    for line in args.infile:
        print ' '.join(w for w in line.split() if w not in filter_out and (voc is None or w in voc))
