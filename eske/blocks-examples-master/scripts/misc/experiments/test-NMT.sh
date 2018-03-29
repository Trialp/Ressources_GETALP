#!/usr/bin/env bash

main_dir=/home/aberard/experiments/DeepPE
data_dir=/home/aberard/experiments/data

src_ext=fr
trg_ext=en
corpus=${data_dir}/news/news

dev_corpus=${corpus}.dev
test_corpus=${corpus}.test
train_corpus=${corpus}.train

output_dir=${main_dir}/$(basename ${corpus})_${src_ext}-${trg_ext}

cd ${main_dir}

scripts/prepare-data.py ${train_corpus}.${src_ext} ${train_corpus}.${trg_ext} ${dev_corpus}.${src_ext} ${dev_corpus}.${trg_ext} -o ${output_dir}
python -m machine_translation.main ${output_dir} ${output_dir} # wait something like 5-7 days (until BLEU score stabilizes on dev set)

# Test
python -m machine_translation.decode ${output_dir}/ ${output_dir}/model < ${test_corpus}.${src_ext} > ${test_corpus}.nmt

echo "## BASELINE ##"
multi-bleu.perl ${test_corpus}.${trg_ext} < ${test_corpus}.mt
echo "## NMT ##"
multi-bleu.perl ${test_corpus}.${trg_ext} < ${test_corpus}.nmt

# TODO: compare with GroundHog