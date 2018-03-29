#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
import sys
import os
import logging

logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO)

threads = 16

commands = """
$MOSES_DIR/scripts/training/mert-moses.pl {corpus}.{src_ext} {corpus}.{trg_ext} $MOSES_DIR/bin/moses {cfg} --mertdir $MOSES_DIR/bin/ --decoder-flags="-threads {threads}" &> {output}
mv mert-work/moses.ini {cfg}.tuned
rm -rf mert-work
""".strip()

if __name__ == '__main__':
    try:
        cfg, corpus, src_ext, trg_ext, output = sys.argv[1:]
    except ValueError:
        sys.exit('Usage: {0} CFG CORPUS SRC_EXT TRG_EXT LOGFILE'.format(sys.argv[0]))

    args = dict(locals())
    commands = commands.strip().format(**args)

    for cmd in commands.split('\n'):
        logging.info(cmd)
        os.system(cmd)
