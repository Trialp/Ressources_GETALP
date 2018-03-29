time word2vec -train ep7.lc.fr -output model.bin.ep7.lc.fr -binary 1 -save-vocab vocab.ep7.lc.fr -cbow 0 -size 100 -window 10 -negative 25 -threads 24
time word2vec -train ep7.lc.en -output model.bin.ep7.lc.en -binary 1 -save-vocab vocab.ep7.lc.en -cbow 0 -size 100 -window 10 -negative 25 -threads 24
