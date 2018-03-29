#!/usr/bin/env python2
from __future__ import division
import subprocess
import sys
import os
import logging

logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO)

help_msg = """
Spawns a numbers of moses jobs to translate files created with the `split.py` script.

CFG:       path to moses.ini
DIR:       directory containing the file splits
FROM:      name of the starting split (e.g., 0 or 15)
JOBS:      number of processes to spawn"""

if __name__ == '__main__':
    try:
        cfg, directory, i, jobs = sys.argv[1:]
        i = int(i)
        jobs = int(jobs)
    except:
        sys.exit('Usage: {0} CFG DIR FROM JOBS\n{1}'.format(sys.argv[0], help_msg))

    processes = []
    for j in range(jobs):
        input_filename = os.path.join(directory, str(i + j))
        output_filename = os.path.join(directory, str(i + j) + '.out')

        cmd = '$MOSES_DIR/bin/moses -f {0} -threads 1 < {1} > {2} 2> /dev/null'.format(cfg, input_filename, output_filename)
        logging.info(cmd)
        p = subprocess.Popen(cmd, shell=True)
        processes.append(p)

    for p in processes:
        p.wait()
