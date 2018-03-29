#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
import sys
import re
from itertools import islice, takewhile
from os.path import dirname, realpath
# allows imports from parent directory
sys.path.append(dirname(dirname(realpath(__file__))))
from utils import uopen, uprint
from trajectories import State

"""
Extract trajectories/alignments from IBM alignment file or TERCOM alignment

Trajectories are sequences of transitions:
    state, new state, action
Outputs a transition per line, the first number is the index of the trajectory the transition belongs to.
The fields of a transition are separated by '|||'.

Actions are:
        SUB index word
        DEL index
        INS index word
        MOVE index position
        STOP
"""

"""
Main program, used as a script to compute TER/IBM trajectories.
"""


def get_shifts(shifts, hyp, hyp_after_shift):
    new_hyp = list(hyp)
    shift_indices = []

    for start, end, dest, phrase in shifts:
        new_hyp = new_hyp[:start] + new_hyp[end + 1:]

        if dest != end:
            dest += 1
            if dest > end:
                dest -= end - start + 1

        shift_indices.append((start, end + 1, dest))
        new_hyp[dest:dest] = phrase.split()

    assert new_hyp == hyp_after_shift
    return shift_indices


def extract_from_ter(ter_filename, src_filename):
    """
    Yields trajectories (lists of operations):
        SUB index word
        DEL index
        INS index word
        MOVE index position
    operations are applied from left to right (indices refer to the current state of the hypothesis)
    """
    with uopen(ter_filename) as f, uopen(src_filename) as src_file:
        while True:
            src_sent = next(src_file).strip()

            ops = []
            lines = list(takewhile(lambda line: line.strip(), f))
            if not lines:
                break
            ref = re.match(r'Original Ref:\s*(.*?)\n', lines[1]).group(1)
            hyp = re.match(r'Original Hyp:\s*(.*?)\n', lines[2]).group(1)
            hyp_after_shift = re.match(r'Hyp After Shift:\s*(.*?)\n', lines[3]).group(1)
            align = re.match(r'Alignment:\s*\((.*?)\)', lines[4]).group(1)

            numshifts = int(re.match(r'NumShifts: (\d+)', lines[5]).group(1))
            regex = re.compile(r'\s*\[(\d+), (\d+), .*?/(.*?)\] \(\[(.*?)\]\)')
            shifts = [regex.match(lines[6 + i]).groups() for i in range(numshifts)]
            shifts = [(int(i), int(j), int(k), re.sub(r',\s+', ' ', words)) for i, j, k, words in shifts]

            shift_indices = get_shifts(shifts, hyp.split(), hyp_after_shift.split())

            for i, j, k in shift_indices:
                l = j - i
                for x in range(l):
                    if k >= i:
                        op = ('MOVE', i, k + l - 1)
                    else:
                        op = ('MOVE', i + x, k + x)
                    ops.append(op)

            ref_iter = iter(ref.split())
            hyp_iter = iter(hyp_after_shift.split())
            i = 0
            for op in align:
                # insert and delete are reversed in TERCOM
                if op != 'D':
                    next(hyp_iter)
                if op != 'I':
                    inserted = next(ref_iter)

                if op == 'S':
                    ops.append(('SUB', i, inserted))
                elif op == 'D':
                    ops.append(('INS', i, inserted))
                elif op == 'I':
                    ops.append(('DEL', i))
                    i -= 1

                i += 1

            ops.append(('STOP',))

            # try to reconstruct reference
            state = State(src_sent, hyp)
            for op in ops:
                state = state.transition(op)

            if state.trg != ref:  # in some weird and rare cases (likely due to a bug in TERCOM)
                yield (src_sent, hyp), []  # empty trajectory (index is skipped in the output)
                continue

            #assert(state.trg == ref)

            yield (src_sent, hyp), ops


regex_IBM_model = re.compile(r'([^\s]*?) \(\{ (.*?)\}\)', re.UNICODE)


def get_pairs(alignment):
    pairs = [(i - 1, int(j) - 1)
             for i, (w, indices) in enumerate(regex_IBM_model.findall(alignment))
             for j in indices.split() if w != 'NULL']
    words = [w for w, _ in regex_IBM_model.findall(alignment)]
    return pairs, words[1:]


def extract_from_ibm(filename1, filename2):
    """
    Takes two GIZA++ alignment file (first file is source->target, second file is target->source),
    and yields trajectories.
    """
    with uopen(filename1) as file1, uopen(filename2) as file2:
        for line1, line2 in zip(islice(file1, 2, None, 3), islice(file2, 2, None, 3)):
            pairs1, words1 = get_pairs(line1.strip())
            pairs2, words2 = get_pairs(line2.strip())
            pairs2 = [pair[::-1] for pair in pairs2]
            sentence = ' '.join(words1)

            intersection = list(set(pairs1).intersection(set(pairs2)))
            ops = []
            pairs = intersection[:]
            pairs.append((len(words1), len(words2)))
            i = 0
            while i < len(words2):
                x = next((x for x, y in pairs if y == i), None)

                if not any(s == i for s, _ in pairs):
                    ops.append(('DEL', i))
                    del words1[i]
                    pairs = [(k - int(i < k), l) for k, l in pairs]
                elif x == i:
                    if words1[i] != words2[i]:
                        ops.append(('SUB', i, words2[i]))
                    pairs.remove((i, i))
                    words1[i] = words2[i]
                    i += 1
                elif x is not None:
                    # move
                    # TODO: fix
                    ops.append(('MOVE', x, i))
                    words1.insert(i, words1.pop(x))
                    pairs = [(i, l) if k == x and l == i else (k + int(i <= k < x), l) for k, l in pairs]
                else:
                    # insertion
                    ops.append(('INS', i, words2[i]))
                    words1.insert(i, words2[i])
                    pairs = [(k + int(i <= k), l) for k, l in pairs]
                    i += 1

            ops.append(('STOP',))

            yield sentence, ops


def full_trajectories(partial_trajectories):
    for k, ((src, trg), actions) in enumerate(partial_trajectories):
        state = State(src, trg)
        for action in actions:
            new_state = state.transition(action)
            yield k, state.src, state.trg, new_state.trg, action
            state = new_state


def main():
    """
    Example usage:
        python -m scripts.compute_trajectories data/eolss-train.pra data/eolss-train.fr > data/eolss-train.trajectories.txt
    """
    try:
        align_file, src_file = sys.argv[1:]
    except ValueError:
        sys.exit('Usage: {} ALIGN_FILE SRC_FILE > OUTPUT_FILE'.format(sys.argv[0]))

    trajectories = extract_from_ter(align_file, src_file)

    for idx, src, trg1, trg2, action in full_trajectories(trajectories):
        uprint(u'|||'.join([str(idx), src, trg1, trg2, ' '.join(map(unicode, action))]))


if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, IOError):
        sys.exit()
