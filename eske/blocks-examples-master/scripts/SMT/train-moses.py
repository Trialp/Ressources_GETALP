#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
import sys
import os
import logging

logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO)

threads = 16

commands = """
rm -rf "{dest}"
$MOSES_DIR/scripts/training/train-model.perl -root-dir "{dest}" -corpus {corpus} -f {src_ext} -e {trg_ext} -alignment grow-diag-final-and -reordering msd-bidirectional-fe -lm 0:3:{lm_corpus}.blm.{trg_ext}:8 -external-bin-dir $GIZA_DIR -mgiza -mgiza-cpus {threads} -cores {threads} --parallel
"""

if __name__ == '__main__':
    try:
        dest, corpus, lm_corpus, src_ext, trg_ext = sys.argv[1:]
    except ValueError:
        sys.exit('Usage: {} DEST_DIR CORPUS LM_CORPUS SRC_EXT TRG_EXT'.format(sys.argv[0]))

    args = dict(locals())
    commands = commands.strip().format(**args)

    for cmd in commands.split('\n'):
        logging.info(cmd)
        os.system(cmd)
