import word2vec
import numpy as np
from repoze.lru import lru_cache


class Model(object):
    def __init__(self, filename):
        self.bivec_model = word2vec.Bivec()
        self.bivec_model.load(filename)

    @lru_cache(maxsize=1024)
    def source_paragraph_vector(self, src_sent):
        try:
            return self.bivec_model.trg_model.sent_vec(src_sent) # FIXME
        except ValueError:
            return None

    @lru_cache(maxsize=1024)
    def target_paragraph_vector(self, trg_sent):
        try:
            return self.bivec_model.src_model.sent_vec(trg_sent)
        except ValueError:
            return None

    def paragraph_vector(self, state):
        src_vector = self.source_paragraph_vector(state.src)
        trg_vector = self.source_paragraph_vector(state.trg)
        if src_vector is None or trg_vector is None:
            return None
        #return src_vector + trg_vector
        #return trg_vector
        return np.concatenate([src_vector, trg_vector])
