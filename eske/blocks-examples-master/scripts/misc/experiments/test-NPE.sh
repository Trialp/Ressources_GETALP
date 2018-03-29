#!/usr/bin/env bash -e

main_dir=/home/aberard/experiments/DeepPE
data_dir=/home/aberard/experiments/data

src_ext=mt
trg_ext=en
corpus=${data_dir}/news/news

dev_corpus=${corpus}.dev
test_corpus=${corpus}.test
train_corpus=${corpus}.train
ls ${dev_corpus} ${test_corpus} ${train_corpus}

output_dir=${main_dir}/$(basename ${corpus})_${src_ext}-${trg_ext}

cd ${main_dir}

scripts/prepare-data.py ${train_corpus}.${src_ext} ${train_corpus}.${trg_ext} ${dev_corpus}.${src_ext} ${dev_corpus}.${trg_ext} -o ${output_dir}
python -m machine_translation.main ${output_dir} ${output_dir}/model

# Test
python -m machine_translation.decode ${output_dir}/ ${output_dir}/model < ${test_corpus}.${src_ext} > ${test_corpus}.nmt

echo "## BASELINE ##"
multi-bleu.perl ${test_corpus}.${trg_ext} < ${test_corpus}.${src_ext}
echo "## NMT ##"
multi-bleu.perl ${test_corpus}.${trg_ext} < ${test_corpus}.nmt
