import os
import pandas as pd
import numpy as np

HOME_PATH = os.path.expanduser('~')

class Gibbs_LDA(object):

    def __init__(self, K, iter_count, alpha, eta):
        """
        Arguments:
        corpus: Collection of (mass, intensity) metabolite tuples;
        K: Number of topics;
        alpha: Hyperparameter for prior on weight vectors theta;
        eta: Hyperparameter for prior on topics beta.
        """
        # self._read_vocab()
        # self._read_corpus()
        self._iter_count = iter_count
        self._K = K
        self._alpha = alpha
        self._eta = eta
        # self._V = len(self.vocab)
        # self._M = len(self.corpus)

    def fit(self, corpus):
        self._fit(corpus)

    def _fit(self, corpus):
        self._initialise(corpus)

    def _initialise(self, corpus):
        self._D, self._W = corpus.shape
        self._total_frequencies = corpus.sum()

        self._zw_counts = np.zeros((self._K, self._W), dtype=np.int)
        self._dz_counts = np.zeros((self._D, self._K), dtype=np.int)
        self._z_counts = np.zeros(self._K, dtype=np.int)
        self._z_history = []

        for d, doc in enumerate(corpus):

            # self._z_history[d] = []
            self._z_history.append([])
            for w, word in enumerate(doc):
                word_frequency = int(doc[word])
                for _ in range(word_frequency):
                    z = np.random.randint(self._K)
                    self._z_history[d].append(z)
                    self._zw_counts[z, w] += 1
                    self._dz_counts[d, z] += 1
                    self._z_counts[z] += 1
            print len(self._z_history[d])

def main():
    K = 10
    corpus = pd.read_pickle(HOME_PATH + '/Projects/tminm/heavy_pickles/preprocessed_corpus.pickle')

    gibbs_lda = Gibbs_LDA(K, iter_count=1000, alpha=1./K, eta=1./K)
    gibbs_lda.fit(corpus)



if __name__ == '__main__':
    main()