#!/usr/bin/env bash

# Builds a simulated PE corpus by translating the source, and using the target as PE reference
# The MT model must have been trained beforehand using `train-europarl.sh`

main_dir=/home/aberard/experiments
data_dir=${main_dir}/data
moses_model_dir=${main_dir}/moses-models
src_ext=fr
trg_ext=en
corpus=${data_dir}/news2/news
dev_corpus=${corpus}.dev
test_corpus=${corpus}.test
train_corpus=${corpus}.train
output_dir=$(dirname ${corpus})/split
splits=128
cfg_file=${moses_model_dir}/europarl_fr-en/binarized/moses.ini.compact

# data preparation
prepare.sh ${corpus}.${src_ext} ${src_ext} > ${corpus}.tok.${src_ext}
prepare.sh ${corpus}.${trg_ext} ${trg_ext} > ${corpus}.tok.${trg_ext}
clean-corpus.py ${corpus}.tok ${corpus}.clean 3 100 ${src_ext} ${trg_ext}
shuf-parallel.py ${corpus}.clean.${src_ext} ${corpus}.clean.${trg_ext}
mv ${corpus}.clean.${src_ext}.shuf ${corpus}.clean.${src_ext}
mv ${corpus}.clean.${trg_ext}.shuf ${corpus}.clean.${trg_ext}

mkdir -p ${output_dir}
split-n.py ${corpus}.clean.${src_ext} ${output_dir} ${splits}

# run instances of moses
ssh bach1 moses-parallel.py ${cfg_file} ${output_dir} 0 16 &
ssh bach2 moses-parallel.py ${cfg_file} ${output_dir} 16 16 &
ssh bach3 moses-parallel.py ${cfg_file} ${output_dir} 32 16 &
ssh bach4 moses-parallel.py ${cfg_file} ${output_dir} 48 16 &
ssh bach5 moses-parallel.py ${cfg_file} ${output_dir} 64 16 &
ssh brahms2 moses-parallel.py ${cfg_file} ${output_dir} 80 8 &
ssh brahms3 moses-parallel.py ${cfg_file} ${output_dir} 88 8 &
ssh hyperion moses-parallel.py ${cfg_file} ${output_dir} 96 16 &
ssh dvorak0 moses-parallel.py ${cfg_file} ${output_dir} 112 16 &

ls ${output_dir} | grep \.out | sort -n | sed -e s@^@${output_dir}/@ | xargs cat > ${corpus}.clean.mt

split-dev-test-train.py ${corpus}.clean ${corpus} 2000 10000 ${src_ext} ${trg_ext} mt
rm -rf ${output_dir}