#!/usr/bin/env bash

main_dir=/home/aberard/experiments/DeepPE
data_dir=/home/aberard/experiments/data

src_ext=fr
trg_ext=en
corpus=${data_dir}/news/news
#lm_corpus=${data_dir}/news-mono/news.2013.en.shuffled
lm_corpus=${data_dir}/commoncrawl/commoncrawl

prepare.sh ${lm_corpus}.${src_ext} ${src_ext} > ${corpus}.tok.${src_ext}
prepare.sh ${lm_corpus}.${trg_ext} ${trg_ext} > ${corpus}.tok.${trg_ext}

clean-corpus.py ${lm_corpus}.tok ${lm_corpus}.clean 3 0 ${src_ext} ${trg_ext}  # remove too short lines
# shuffle
shuf-parallel.py ${lm_corpus}.clean.${src_ext} ${lm_corpus}.clean.${trg_ext}
mv ${lm_corpus}.clean.${src_ext}.shuf ${lm_corpus}.clean.${src_ext}
mv ${lm_corpus}.clean.${trg_ext}.shuf ${lm_corpus}.clean.${trg_ext}

multivec-bi --dimension 620 --min-count 10 --window-size 8 --threads 16 --iter 20 --negative 10 --beta 2 --train-src ${lm_corpus}.clean.${src_ext} \
    --train-trg ${lm_corpus}.clean.${trg_ext} --save ${lm_corpus}.multivec.${src_ext}-${trg_ext}.bin
multivec-bi --load ${lm_corpus}.multivec.${src_ext}-${trg_ext}.bin --save-src ${lm_corpus}.multivec.${src_ext}.bin --save-trg ${lm_corpus}.multivec.${trg_ext}.bin
multivec-mono --load ${lm_corpus}.multivec.${src_ext}.bin --save-vectors ${lm_corpus}.vectors.${src_ext}.bin
multivec-mono --load ${lm_corpus}.multivec.${trg_ext}.bin --save-vectors ${lm_corpus}.vectors.${trg_ext}.bin


dev_corpus=${corpus}.dev
test_corpus=${corpus}.test
train_corpus=${corpus}.train

output_dir=${main_dir}/$(basename ${corpus})_${src_ext}-${trg_ext}

cd ${main_dir}

scripts/prepare-data.py ${train_corpus}.${src_ext} ${train_corpus}.${trg_ext} ${dev_corpus}.${src_ext} ${dev_corpus}.${trg_ext}
python -m machine_translation.main ${output_dir} ${output_dir} # wait something like 5-7 days (until BLEU score stabilizes on dev set)

# Test
python -m machine_translation.decode ${output_dir} ${output_dir} < ${test_corpus}.${src_ext} > ${test_corpus}.nmt

echo "## BASELINE ##"
multi-bleu.perl ${test_corpus}.${trg_ext} < ${test_corpus}.mt
echo "## NMT ##"
multi-bleu.perl ${test_corpus}.${trg_ext} < ${test_corpus}.nmt

# TODO: compare with GroundHog