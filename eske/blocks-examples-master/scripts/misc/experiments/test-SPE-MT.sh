#!/usr/bin/env bash

# Instead of translating from target language to target language, translate from scratch
# from source language to target language.

main_dir=/home/aberard/experiments
data_dir=${main_dir}/data
moses_model_dir=${main_dir}/moses-models

src_ext=fr
trg_ext=en
corpus=${data_dir}/news/news

dev_corpus=${corpus}.dev
test_corpus=${corpus}.test
train_corpus=${corpus}.train
output_dir=${moses_model_dir}/$(basename ${corpus})_${src_ext}-${trg_ext}

train-lm.py ${train_corpus} ${trg_ext}

## Training baseline SPE model ##

train-moses.py ${output_dir} ${train_corpus} ${train_corpus} ${src_ext} ${trg_ext}
tune-moses.py ${output_dir}/model/moses.ini ${dev_corpus} ${src_ext} ${trg_ext} ${output_dir}/tuning.out

${MOSES_DIR}/bin/moses -threads 16 -f ${output_dir}/model/moses.ini.tuned < ${test_corpus}.${src_ext} > ${test_corpus}.pe-mt

echo "## BASELINE ##"
multi-bleu.perl ${test_corpus}.${trg_ext} < ${test_corpus}.mt
echo "## SPE-MT ##"
multi-bleu.perl ${test_corpus}.${trg_ext} < ${test_corpus}.pe-mt