#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
from tempfile import NamedTemporaryFile, mktemp
from subprocess import check_output, call
from utils import uopen
import os
import re
import codecs


def tercom(hypotheses, references):
    """
    Computes the TER between hypotheses and references.
    :param hypotheses: [str]
    :param references: [str]
    :return: float
    """
    with NamedTemporaryFile('w') as hypothesis_file, NamedTemporaryFile('w') as reference_file:
        for i, (hypothesis, reference) in enumerate(zip(hypotheses, references)):
            hypothesis_file.write('{} ({})\n'.format(hypothesis, i))
            reference_file.write('{} ({})\n'.format(reference, i))
        hypothesis_file.flush()
        reference_file.flush()

        cmd = ['java', '-jar', 'tercom.jar', '-h', hypothesis_file.name, '-r', reference_file.name]
        output = check_output(cmd)
        error = re.findall(r'Total TER: (.*?) ', output, re.MULTILINE)[0]
        return float(error)


def tercom_unicode(hypotheses, references):
    """
    Computes the TER between hypotheses and references.
    :param hypotheses: [unicode]
    :param references: [unicode]
    :return: float
    """
    writer = codecs.getwriter('utf-8')
    with NamedTemporaryFile('w') as hypothesis_file, NamedTemporaryFile('w') as reference_file:
        hypothesis_file = writer(hypothesis_file)
        reference_file = writer(reference_file)
        for i, (hypothesis, reference) in enumerate(zip(hypotheses, references)):
            hypothesis_file.write(u'{} ({})\n'.format(hypothesis, i))
            reference_file.write(u'{} ({})\n'.format(reference, i))
        hypothesis_file.flush()
        reference_file.flush()

        cmd = ['java', '-jar', 'tercom.jar', '-h', hypothesis_file.name, '-r', reference_file.name]
        output = check_output(cmd)
        error = re.findall(r'Total TER: (.*?) ', output, re.MULTILINE)[0]
        return float(error)


def tercom_scores(hypotheses, references):
    """
    Returns a list of TERCOM scores
    """
    with NamedTemporaryFile('w') as hypothesis_file, NamedTemporaryFile('w') as reference_file:
        for i, (hypothesis, reference) in enumerate(zip(hypotheses, references)):
            hypothesis_file.write('{} ({})\n'.format(hypothesis, i))
            reference_file.write('{} ({})\n'.format(reference, i))
        hypothesis_file.flush()
        reference_file.flush()

        filename = mktemp()

        cmd = ['java', '-jar', 'tercom.jar', '-h', hypothesis_file.name, '-r', reference_file.name,
               '-o', 'ter', '-n', filename]
        output = open('/dev/null', 'w')
        call(cmd, stdout=output, stderr=output)

    with open(filename + '.ter') as f:
        lines = list(f)
        scores = [float(line.split(' ')[-1]) for line in lines[2:]]

    os.remove(filename + '.ter')
    return scores


def tercom_scores_unicode(hypotheses, references):
    """
    Returns a list of TERCOM scores
    """
    writer = codecs.getwriter('utf-8')
    with NamedTemporaryFile('w') as hypothesis_file, NamedTemporaryFile('w') as reference_file:
        hypothesis_file = writer(hypothesis_file)
        reference_file = writer(reference_file)

        for i, (hypothesis, reference) in enumerate(zip(hypotheses, references)):
            hypothesis_file.write(u'{} ({})\n'.format(hypothesis, i))
            reference_file.write(u'{} ({})\n'.format(reference, i))

        hypothesis_file.flush()
        reference_file.flush()

        filename = mktemp()

        cmd = ['java', '-jar', 'tercom.jar', '-h', hypothesis_file.name, '-r', reference_file.name,
               '-o', 'ter', '-n', filename]
        output = uopen('/dev/null', 'w')
        call(cmd, stdout=output, stderr=output)

    with uopen(filename + '.ter') as f:
        lines = list(f)
        scores = [float(line.split(' ')[-1]) for line in lines[2:]]

    os.remove(filename + '.ter')
    return scores
