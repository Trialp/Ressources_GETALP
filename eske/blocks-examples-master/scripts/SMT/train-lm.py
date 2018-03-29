#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
import sys
import os
import logging

logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO)

commands = """
$MOSES_DIR/bin/lmplz -o {order} < {lm_corpus}.{trg_ext} > {lm_corpus}.arpa.{trg_ext}
$MOSES_DIR/bin/build_binary {lm_corpus}.arpa.{trg_ext} {lm_corpus}.blm.{trg_ext}
""".strip()

if __name__ == '__main__':
    try:
        lm_corpus, trg_ext = sys.argv[1:3]
        order = int(sys.argv[3]) if len(sys.argv) >= 4 else 3
    except ValueError:
        sys.exit('Usage: {} CORPUS EXTENSION [ORDER]'.format(sys.argv[0]))

    args = dict(locals())
    commands = commands.strip().format(**args)

    for cmd in commands.split('\n'):
        logging.info(cmd)
        os.system(cmd)
