#!/usr/bin/env bash

# TODO: train baseline MT system on Europarl

main_dir=/home/aberard/experiments
data_dir=${main_dir}/data
moses_model_dir=${main_dir}/moses-models

## Building PE corpus ##

# TODO: filter corpus to remove duplicate sentences
#train_corpus=${data_dir}/EMEA/EMEA.train
#test_corpus=${data_dir}/EMEA/EMEA.test
#dev_corpus=${data_dir}/EMEA/EMEA.dev

train_corpus=${data_dir}/news/news.train
test_corpus=${data_dir}/news/news.test
dev_corpus=${data_dir}/news/news.dev
# TODO: clean corpus to remove empty lines

src_ext=fr
mt_ext=mt
trg_ext=en

# Specifically for EMEA
if [ $(basename ${test_corpus}) -eq EMEA.test ] ; then
head -n 1000 ${test_corpus}.${mt_ext} > ${dev_corpus}.${mt_ext}
head -n 1000 ${test_corpus}.${trg_ext} > ${dev_corpus}.${trg_ext}
sed -n '1001,3000p' ${test_corpus}.${mt_ext} > ${test_corpus}.sample.${mt_ext}
sed -n '1001,3000p' ${test_corpus}.${trg_ext} > ${test_corpus}.sample.${trg_ext}
test_corpus=${test_corpus}.sample
fi

#train-lm.py ${train_corpus} ${trg_ext}

output_dir=${moses_model_dir}/$(basename ${train_corpus})_${mt_ext}-${trg_ext}

## Training baseline SPE model ##

#tuning_log_file=${moses_model_dir}/tuning.out

#train-moses.py ${output_dir} ${train_corpus} ${train_corpus} ${mt_ext} ${trg_ext}
#tune-moses.py ${output_dir}/model/moses.ini ${dev_corpus} ${mt_ext} ${trg_ext} ${tuning_log_file}

## Testing model ##

test_output=${moses_model_dir}/$(basename ${test_corpus}).out
${moses_dir}/bin/moses -f ${output_dir}/model/moses.ini.tuned < ${test_corpus}.${mt_ext} > ${test_corpus}.pe
multi-bleu.perl ${test_corpus}.${trg_ext} < ${test_corpus}.pe > ${test_output}

## SPE model with large language model: TODO
## Baseline NPE
# TODO: frequent checkpoints, with frequent evaluation on small dev corpus

deep_pe_dir=${main_dir}/DeepPE
${deep_pe_dir}/scripts/prepare-data ${train_corpus}.${src_ext} ${train_corpus}.${mt_ext} ${train_corpus}.${mt_ext} ${dev_corpus}.${src_ext} ${dev_corpus}.${mt_ext} ${dev_corpus}.${mt_ext} -o ${deep_pe_dir}/data

## Baseline NPE + pre-trained word embeddings
## Extended NPE