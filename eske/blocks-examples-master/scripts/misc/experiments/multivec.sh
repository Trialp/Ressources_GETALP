#!/usr/bin/env bash

main_dir=/home/aberard/experiments
data_dir=${main_dir}/data
ext=en

corpus=${data_dir}/news-mono/news.2013
corpus_name=$(basename ${corpus})
output_dir=${main_dir}/multivec-models

# normalize punctuation, tokenize and lowercase
prepare.sh ${corpus}.${ext} ${ext} > ${corpus}.tok.${ext}

# remove too long and too short lines
clean-corpus.py ${corpus}.tok ${corpus}.clean 3 0 ${ext}

# shuffle
shuf-parallel.py ${corpus}.clean.${ext}
mv ${corpus}.clean.${ext}.shuf ${corpus}.clean.${ext}
# remove sentences that are also in training data
minus.py ${corpus}.clean.${ext} ${data_dir}/news/news.clean.${ext} > ${corpus}.filtered.${ext}

multivec-mono --dimension 620 --iter 20 --min-count 10 --negative 5 --subsampling 1e-04 --window-size 5 \
--alpha 0.05 --train ${corpus}.filtered.${ext} --save ${output_dir}/${corpus_name}.multivec.${ext}.bin --threads 16




# Bilingual
src_ext=fr
trg_ext=en

corpus=${data_dir}/commoncrawl/commoncrawl
corpus_name=$(basename ${corpus})
output_dir=${main_dir}/multivec-models

# normalize punctuation, tokenize and lowercase
prepare.sh ${corpus}.${src_ext} ${src_ext} > ${corpus}.tok.${src_ext}
prepare.sh ${corpus}.${trg_ext} ${trg_ext} > ${corpus}.tok.${trg_ext}

# remove too long and too short lines
clean-corpus.py ${corpus}.tok ${corpus}.clean 3 0 ${src_ext} ${trg_ext}

# shuffle
shuf-parallel.py ${corpus}.clean.${src_ext} ${corpus}.clean.${trg_ext}
mv ${corpus}.clean.${src_ext}.shuf ${corpus}.clean.${src_ext}
mv ${corpus}.clean.${trg_ext}.shuf ${corpus}.clean.${trg_ext}
# remove sentences that are also in training data
#minus.py ${corpus}.clean.${ext} ${data_dir}/news/news.clean.${ext} > ${corpus}.filtered.${ext}

multivec-bi --dimension 620 --iter 20 --min-count 10 --negative 5 --subsampling 1e-04 --window-size 5 --alpha 0.05 \
 --train-src ${corpus}.filtered.${src_ext} --train-trg ${corpus}.filtered.${trg_ext} \
 --save ${output_dir}/${corpus_name}.multivec.${src_ext}-${trg_ext}.bin --threads 16
multivec-bi --load ${output_dir}/${corpus_name}.multivec.${src_ext}-${trg_ext}.bin \
--save-src ${output_dir}/${corpus_name}.multivec.${src_ext}.bi \
--save-trg ${output_dir}/${corpus_name}.multivec.${trg_ext}.bi