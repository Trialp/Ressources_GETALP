#!/usr/bin/env bash -e

# Run experiment similar to Bahdanau et al., 2014 (http://arxiv.org/pdf/1409.0473v6.pdf)
# Neural machine translation with attention model (RNNsearch-50 model), English to French
# on WMT 14 data, filtered by Axelrod et al., 2011

main_dir=/home/aberard/experiments/blocks-examples    # change this

scripts=${main_dir}/scripts
data_dir=${main_dir}/data/WMT14
model_dir=${data_dir}/model

cd ${main_dir}
mkdir -p ${data_dir}
mkdir -p ${model_dir}

cd ${data_dir}
wget http://www-lium.univ-lemans.fr/~schwenk/cslm_joint_paper/data/bitexts.tgz
tar xzf bitexts.tgz
rm bitexts.tgz
gunzip bitexts.selected/*
cat bitexts.selected/*.en > WMT14.en
cat bitexts.selected/*.fr > WMT14.fr
rm -rf bitexts.selected

# Files are already tokenized, (Bahdanau et al., 2014) don't do any special preprocessing (no lowercasing etc.)
#lowercase.perl < WMT14.en > WMT14.tok.en
#mv WMT14.tok.en WMT14.en
#lowercase.perl < WMT14.fr > WMT14.tok.fr
#mv WMT14.tok.fr WMT14.fr

${scripts}/shuf-corpus.py WMT14 fr en    # uses a lot of memory

wget http://www-lium.univ-lemans.fr/~schwenk/cslm_joint_paper/data/dev+test.tgz
tar xzf dev+test.tgz
rm dev+test.tgz
chmod -R 664 dev/*
mv dev/ntst14.en WMT14.test.en
mv dev/ntst14.fr WMT14.test.fr
mv dev/ntst1213.en WMT14.dev.en
mv dev/ntst1213.fr WMT14.dev.fr
rm -rf dev

${scripts}/shuf-corpus.py WMT14.test fr en
${scripts}/shuf-corpus.py WMT14.dev fr en

head -n3000 WMT14.dev.en > WMT14.dev.sample.en   # 6000 lines is too much
head -n3000 WMT14.dev.fr > WMT14.dev.sample.fr

# translate from English to French
${scripts}/prepare-data.py WMT14.en WMT14.fr WMT14.dev.sample.en WMT14.dev.sample.fr -o . --sample-size 1500

cd ${main_dir}
alias python2="THEANO_FLAGS=device=gpu0,floatX=float32,on_unused_input=warn python2"
python2 -m machine_translation ${data_dir} ${model_dir} --reload 0 --val-burn-in 50000 |& tee -a ${model_dir}/log.txt
