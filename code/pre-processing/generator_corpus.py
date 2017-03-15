
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
    def __init__(self, alphas, beta, xi, T):
        self.alphas = alphas
        self.beta = beta
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

    def generate_corpus(self):
        corpus = {}
        for d in range(self.T):
            corpus[d] = {}
            N = np.random.poisson(self.xi)
            for w in range(N):
                topic_distrib = np.random.multinomial(1, self.alphas[d])
                z = np.where(topic_distrib == 1)[0][0]
                word_distrib = np.random.multinomial(1, self.beta[z])
                w = np.where(word_distrib == 1)[0][0]
                w_key = self.vocab[w]
                if w_key not in corpus[d]:
                    corpus[d][w_key] = 0
                corpus[d][w_key] += 1
        return corpus