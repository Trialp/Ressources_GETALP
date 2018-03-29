#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
import sys
from tempfile import NamedTemporaryFile
from subprocess import call


def run_tercom(hypotheses, references, output_filename):
    """
    Computes the TER between hypotheses and references.
    :param hypotheses: [str]
    :param references: [str]
    """
    with NamedTemporaryFile('w') as hypothesis_file, NamedTemporaryFile('w') as reference_file:
        for i, (hypothesis, reference) in enumerate(zip(hypotheses, references)):
            hypothesis_file.write('{} ({})\n'.format(hypothesis, i))
            reference_file.write('{} ({})\n'.format(reference, i))
        hypothesis_file.flush()
        reference_file.flush()

        cmd = ['java', '-jar', 'tercom.jar', '-h', hypothesis_file.name, '-r', reference_file.name,
               '-o', 'pra', '-n', output_filename]
        call(cmd)


if __name__ == '__main__':
    """
    Computes TERCOM alignment of hypothesis file and reference file.

    Example usage:
        ./tercom-alignment.py data/train.mt data/train.pe data/train
    """

    try:
        hyp, ref, out = sys.argv[1:]
    except ValueError:
        sys.exit('Usage: {} HYP_FILE REF_FILE OUTPUT_FILE'.format(sys.argv[0]))

    with open(hyp) as hyp_file, open(ref) as ref_file:
        hypotheses = [line.strip() for line in hyp_file]
        references = [line.strip() for line in ref_file]
        run_tercom(hypotheses, references, out)
