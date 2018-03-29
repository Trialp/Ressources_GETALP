#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
import argparse
import logging
import os
import subprocess
import shutil
from itertools import islice

parser = argparse.ArgumentParser(description="""
Prepares data for NMT. Requires tokenized source and target training and dev files.
Will copy those files under appropriate names in the output directory and compute vocabularies.
""", formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("src_file", type=str, help="Source file")
parser.add_argument("trg_file", type=str, help="Target file")
parser.add_argument("dev_src_file", type=str, help="Dev source file")
parser.add_argument("dev_trg_file", type=str, help="Dev target file")

parser.add_argument('--sample-size', type=int, default=2000, help="Train sample size")
parser.add_argument("--src-vocab", type=int, default=30000,
                    help="Source language vocabulary size")
parser.add_argument("--trg-vocab", type=int, default=30000,
                    help="Target language vocabulary size")
parser.add_argument("-o", "--output-dir", type=str, help="Output directory")


def create_vocabulary(filename, output, preprocess_file, vocab_size):
    logger.info("Creating vocabulary [{}]".format(output))
    if not os.path.exists(output):
        subprocess.check_call(" python {} -d {} -v {} {}".format(
            preprocess_file, output, vocab_size,
            filename),
            shell=True)
    else:
        logger.info("...file exists [{}]".format(output))


def create_vocabularies(output_dir, tr_files, preprocess_file):
    src_vocab_name = os.path.join(output_dir, 'vocab.src.pkl')
    trg_vocab_name = os.path.join(output_dir, 'vocab.trg.pkl')

    src_filename, trg_filename = tr_files

    create_vocabulary(src_filename, src_vocab_name, preprocess_file, args.src_vocab)
    create_vocabulary(trg_filename, trg_vocab_name, preprocess_file, args.trg_vocab)


def head(src_filename, dst_filename, n=10):
    with open(src_filename) as src_file, open(dst_filename, 'w') as dst_file:
        dst_file.writelines(islice(src_file, n))


def main():
    train_files = train_src, train_trg = args.src_file, args.trg_file
    dev_src, dev_trg = args.dev_src_file, args.dev_trg_file
    sample_size = args.sample_size

    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    preprocess_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'preprocess.py')

    bleu_src_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'multi-bleu.perl')
    bleu_file = os.path.join(output_dir, 'multi-bleu.perl')
    shutil.copy(bleu_src_file, bleu_file)

    shutil.copy(train_src, os.path.join(output_dir, 'train.src'))
    shutil.copy(train_trg, os.path.join(output_dir, 'train.trg'))

    # Subset of training corpus for BLEU eval
    head(train_src, os.path.join(output_dir, 'train.sample.src'), sample_size)
    head(train_trg, os.path.join(output_dir, 'train.sample.trg'), sample_size)

    shutil.copy(dev_src, os.path.join(output_dir, 'dev.src'))
    shutil.copy(dev_trg, os.path.join(output_dir, 'dev.trg'))

    # Apply preprocessing and construct vocabularies
    create_vocabularies(output_dir, train_files, preprocess_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('prepare_data')
    args = parser.parse_args()
    main()
