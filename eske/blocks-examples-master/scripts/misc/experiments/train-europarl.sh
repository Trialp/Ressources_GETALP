#!/usr/bin/env bash

# Trains a baseline SMT model on Europarl data

main_dir=/home/aberard/experiments
data_dir=${main_dir}/data
moses_model_dir=${main_dir}/moses-models
src_ext=fr
trg_ext=en
corpus=${data_dir}/europarl/europarl
dev_corpus=${corpus}.dev
test_corpus=${corpus}.test
train_corpus=${corpus}.train
output_dir=${moses_model_dir}/$(basename ${corpus})_${src_ext}-${trg_ext}

# normalize punctuation, tokenize and lowercase
prepare.sh ${corpus}.${src_ext} ${src_ext} > ${corpus}.tok.${src_ext}
prepare.sh ${corpus}.${trg_ext} ${trg_ext} > ${corpus}.tok.${trg_ext}

# remove too long and too short lines
clean-corpus.py ${corpus}.tok ${corpus}.clean 3 80 ${src_ext} ${trg_ext}

# split corpus into train/dev/test
shuf-parallel.py ${corpus}.clean.${src_ext} ${corpus}.clean.${trg_ext}
mv ${corpus}.clean.${src_ext}.shuf ${corpus}.clean.${src_ext}
mv ${corpus}.clean.${trg_ext}.shuf ${corpus}.clean.${trg_ext}

split-dev-test-train.py ${corpus}.clean ${corpus} 2000 5000 ${src_ext} ${trg_ext}

train-lm.py ${train_corpus} ${trg_ext}

## Training baseline SPE model ##

train-moses.py ${output_dir} ${train_corpus} ${train_corpus} ${src_ext} ${trg_ext}
tune-moses.py ${output_dir}/model/moses.ini ${dev_corpus} ${src_ext} ${trg_ext} ${output_dir}/tuning.out

mkdir ${output_dir}/binarized
${MOSES_DIR}/bin/processPhraseTableMin -in ${output_dir}/model/phrase-table.gz -out ${output_dir}/binarized/phrase-table -nscores 4 -threads 16
${MOSES_DIR}/bin/processLexicalTableMin -in ${output_dir}/model/reordering-table.wbe-msd-bidirectional-fe.gz -out ${output_dir}/binarized/reordering-table -threads 16

cat ${output_dir}/model/moses.ini.tuned | sed "s@PhraseDictionaryMemory\(.*path=\)[^ ]*\(.*\)@PhraseDictionaryCompact\1${output_dir}/binarized/phrase-table\2@" \
    > ${output_dir}/binarized/moses.ini.tmp

cat ${output_dir}/binarized/moses.ini.tmp | sed "s@\(LexicalReordering.*path=\).*@\1${output_dir}/binarized/reordering-table@" \
    > ${output_dir}/binarized/moses.ini.compact

rm ${output_dir}/binarized/moses.ini.tmp
