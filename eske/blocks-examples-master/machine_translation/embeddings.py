import struct
import numpy as np
from itertools import takewhile
from stream import get_vocab
from six.moves import cPickle


def load_embeddings(filename, dim, vocab, vocab_size=30000, unk_id=0):
    """
    Extract word embeddings from a word2vec model for each word in the vocabulary. Unknown word embeddings
    are set to zero.
    """
    vocab = get_vocab(vocab, vocab_size, unk_id)
    W = np.zeros((vocab_size, dim), dtype=np.float32)

    with open(filename, 'rb') as f:
        it = iter(lambda: f.read(1), '')

        n, dim_ = map(int, f.readline().split())
        assert dim == dim_, 'Wrong embedding dimension'
        for _ in range(n):
            word = ''.join(takewhile(lambda x: x != ' ', it))
            vec = f.read(dim * 4 + 1)[:-1]
            vec = np.array(struct.unpack('f' * dim, vec), dtype='float')
            if word in vocab:
                W[vocab[word]] = vec

    return W

