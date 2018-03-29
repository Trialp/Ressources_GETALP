#!/usr/bin/env bash -e

# Train an auto-encoder using the RNNsearch model on monolingual news data
# Introduce noise in the input by dropping random words.

main_dir=/home/aberard/experiments/DeepPE
data_dir=/home/aberard/experiments/data
output_dir=${main_dir}/data/mono
model_dir=${output_dir}/model

corpus=${data_dir}/news-mono/news.2013
corpus_name=$(basename ${corpus})

ext=en

${main_dir}/scripts/split-dev-test-train.py ${corpus}.filtered ${corpus} 3000 1500 ${ext}

${main_dir}/scripts/preprocess.py -d ${output_dir}/vocab.src.pkl -v 30000 ${corpus}.train.${ext}
ln -s ${output_dir}/vocab.src.pkl ${output_dir}/vocab.trg.pkl
cp ${corpus}.train.${ext} ${output_dir}/train.src
ln -s ${output_dir}/train.src ${output_dir}/train.trg
cp ${corpus}.dev.${ext} ${output_dir}/dev.src
ln -s ${output_dir}/dev.src ${output_dir}/dev.trg
head -n 1500 ${output_dir}/train.src > ${output_dir}/train.sample.src
ln -s ${output_dir}/train.sample.src ${output_dir}/train.sample.trg
cp ${main_dir}/scripts/multi-bleu.perl ${output_dir}

multivec-mono --dimension 620 --iter 20 --min-count 10 --negative 5 --subsampling 1e-04 --window-size 5 \
--alpha 0.05 --train ${corpus}.filtered.${src_ext} --save-vectors-bin ${output_dir}/vectors.src.bin
ln -s ${output_dir}/vectors.src.bin ${output_dir}/vectors.trg.bin

mkdir ${output_dir}/model

python-gpu0 -m machine_translation.main ${output_dir} ${model_dir} --dropout 0.5 --train-noise 0.4 --load-embeddings 1 --fix-embeddings 0 |& tee -a ${model_dir}/log.txt
