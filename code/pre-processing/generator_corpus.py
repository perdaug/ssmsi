
"""
VERSION
- Python 2

FUNCTION
- Corpus generation
"""

import numpy as np
import string


class Generator_Corpus(object):
    '''
    Terms:
    - pp: pre-processed
    '''
    def __init__(self, alpha_init, theta_init, beta_init, xi, T):
        if alpha_init is not None:
            self.theta = self._softmax_matrix(alpha_init)
        elif theta_init is not None:
            self.theta = theta_init
        self.beta = beta_init
        self.xi = xi
        self.K, self.V = self.beta.shape
        self.vocab = self._generate_vocab()
        self.T = T

    def _generate_vocab(self):
        vocab = []
        for w in range(self.V):
            vocab.append(string.ascii_lowercase[w])
        return np.array(vocab)
# ___________________________________________________________________________
# TODO: Beta is phi

    def generate_corpus(self):
        corpus = {}
        for d in range(self.T):
            corpus[d] = {}
            N = np.random.poisson(self.xi)
            for w in range(N):
                topic_distrib = np.random.multinomial(1, self.theta[d])
                z = np.where(topic_distrib == 1)[0][0]
                word_distrib = np.random.multinomial(1, self.beta[z])
                w = np.where(word_distrib == 1)[0][0]
                w_key = self.vocab[w]
                if w_key not in corpus[d]:
                    corpus[d][w_key] = 0
                corpus[d][w_key] += 1

        return corpus

    def _softmax_matrix(self, arr):
        arr_softmax = np.array(arr)
        for idx_r, row in enumerate(arr):
            arr_softmax[idx_r] = np.exp(row) / np.sum(np.exp(row))
        return arr_softmax
