#!/usr/bin/env bash -e

# Train an auto-encoder using the RNNsearch model on monolingual news data
# Introduce noise in the input by dropping random words.

main_dir=/home/aberard/experiments/blocks-examples    # change this

scripts=${main_dir}/scripts
data_dir=${main_dir}/data/mono
model_dir=${data_dir}/model

cd ${main_dir}
mkdir -p ${data_dir}
mkdir -p ${model_dir}

cd ${data_dir}

# Prepare training data

wget http://www.statmt.org/wmt15/training-monolingual-news-crawl-v2/news.2014.en.shuffled.v2.gz
gunzip news.2014.en.shuffled.v2.gz
mv news.2014.en.shuffled.v2 news.2014.en

${scripts}/normalize-punctuation.perl -l en < news.2014.en | ${scripts}/tokenizer.perl -l en -threads 16 | lowercase.perl > news.2014.tok.en
${scripts}/clean-corpus.py --min 1 --max 0 news.2014.tok news.2014.clean en
rm news.2014.tok*  # remove unnecessary files
${scripts}/shuf-corpus.py news.2014.clean en

# Prepare test/dev data

#${scripts}/split-corpus.py news.2014.clean news --dev-size 3000 --test-size 3000 en

wget http://www.statmt.org/wmt15/dev-v2.tgz
tar xzf dev-v2.tgz
rm dev-v2.tgz
mv dev/newstest2012.en news.test.en
mv dev/newstest2013.en news.dev.en
rm -rf dev

${scripts}/normalize-punctuation.perl -l en < news.test.en | ${scripts}/tokenizer.perl -l en -threads 16 | lowercase.perl > news.test.tok.en
${scripts}/normalize-punctuation.perl -l en < news.dev.en | ${scripts}/tokenizer.perl -l en -threads 16 | lowercase.perl > news.dev.tok.en
${scripts}/clean-corpus.py --min 1 --max 0 news.test.tok news.test.clean en
${scripts}/clean-corpus.py --min 1 --max 0 news.dev.tok news.dev.clean en
rm news.test.tok.en
rm news.dev.tok.en
${scripts}/shuf-corpus.py news.test.clean en
${scripts}/shuf-corpus.py news.dev.clean en

# Prepare NMT

${scripts}/preprocess.py -d vocab.src.pkl -v 30000 news.2014.clean.en
ln -s vocab.src.pkl vocab.trg.pkl

mv news.2014.clean.en train.src
ln -s train.src train.trg
mv news.dev.clean.en dev.src
ln -s dev.src dev.trg
mv news.test.clean.en test.src
ln -s test.src test.trg
cp ${scripts}/multi-bleu.perl .

head -n 1500 train.src > train.sample.src
head -n 1500 train.trg > train.sample.trg

# Train word embeddings

# multivec-mono must be in PATH
multivec-mono --dimension 620 --iter 20 --min-count 10 --negative 5 --subsampling 1e-04 --window-size 5 \
--alpha 0.05 --train train.src --save-vectors-bin vectors.src.bin
ln -s vectors.src.bin vectors.trg.bin

cd ${main_dir}
alias python2="THEANO_FLAGS=device=gpu0,floatX=float32,on_unused_input=warn python2"
python2 -m machine_translation ${data_dir} ${model_dir} --reload 0 --val-burn-in 80000 --train-noise 0.2 --dropout 0.5 --load-embeddings 1 |& tee -a ${model_dir}/log.txt
