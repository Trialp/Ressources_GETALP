#!/usr/bin/env bash -e

# Example of neural machine translation on news commentary

main_dir=/home/aberard/experiments/blocks-examples    # change this
#main_dir=/home/alex/Documents/Programming/blocks-examples

scripts=${main_dir}/scripts
data_dir=${main_dir}/data/news
model_dir=${data_dir}/model

cd ${main_dir}
mkdir -p ${data_dir}
mkdir -p ${model_dir}

cd ${data_dir}
wget http://www.statmt.org/wmt15/training-parallel-nc-v10.tgz -O news.tgz
tar xzf news.tgz
rm news.tgz
#rename s/news-commentary-v10.fr-en/news/ *    # under recent linux
rename news-commentary-v10.fr-en news *
rm news-commentary*

${scripts}/normalize-punctuation.perl -l en < news.en | ${scripts}/tokenizer.perl -l en -threads 16 | ${scripts}/lowercase.perl > news.tok.en
${scripts}/normalize-punctuation.perl -l fr < news.fr | ${scripts}/tokenizer.perl -l fr -threads 16 | ${scripts}/lowercase.perl > news.tok.fr

${scripts}/clean-corpus.py --min 1 --max 0 news.tok news.clean fr en
rm news.tok*  # remove unnecessary files

${scripts}/shuf-corpus.py news.clean fr en
${scripts}/split-corpus.py news.clean news --dev-size 1500 --test-size 3000 fr en

# translate from English to French
${scripts}/prepare-data.py news.train.en news.train.fr news.dev.en news.dev.fr -o . --sample-size 1500

cd ${main_dir}
alias python2="THEANO_FLAGS=device=gpu0,floatX=float32,on_unused_input=warn python2"
python2 -m machine_translation ${data_dir} ${model_dir} --reload 0 --val-burn-in 20000 |& tee -a ${model_dir}/log.txt
