# Implementation of a variational LDA.
# 
# Code refers to the variational LDA implementation of Dr Simon Rogers.
# Comments refer to the LDA paper (Blei et al. 2003)  
# and the the online LDA implementation (Hoffman et al. 2010).

import os
import pandas as pd
import numpy as np
from scipy.special import psi

HOME_PATH = os.path.expanduser('~')
np.set_printoptions(edgeitems=5)

class Variational_LDA(object):

    def __init__(self, iter_count, K, alpha, eta):
        """
        Arguments:
        corpus: Collection of (mass, intensity) metabolite tuples;
        K: Number of topics;
        alpha: Hyperparameter for prior on weight vectors theta;
        eta: Hyperparameter for prior on topics beta.
        """
        self.iter_count = iter_count
        self.K = K
        self.alpha = alpha
        self.eta = eta

    def fit(self, corpus, vocab):
        self._fit(corpus, vocab)

    def get_theta(self):
        return self._gamma / self._gamma.sum(axis=1)[:, None]

    def _fit(self, corpus, vocab):
        self._init_vb(corpus, vocab)

        for it in range(self.iter_count):
            print("Iteration: %s" % it)
            self._vb_step()

    def _init_vb(self, corpus, vocab):
        self.corpus = corpus
        self.vocab = vocab
        self.V = len(vocab)
        self.D = len(corpus)

        self._beta = np.random.rand(self.K, self.V) 
        self._beta /= self._beta.sum(axis=1)[:, None]

        # Initialising Gamma and Phi.
        self._phi = {}
        self._gamma = np.zeros((self.D, self.K))
        for d, doc in enumerate(self.corpus):
            self._phi[doc] = {}
            for word in self.corpus[doc]:
                # Line (1) in Figure 6.
                self._phi[doc][word] = np.full(self.K, 1./self.K)
            total_intensity = sum(self.corpus[doc].values())
            # Line (2) in Figure 6 .
            self._gamma[d, :] = self.alpha + total_intensity / self.K

    def _vb_step(self):
        temp_beta = self._e_step()
        temp_beta += self.eta
        temp_beta /= temp_beta.sum(axis=1)[:, None]
        self._beta = temp_beta

    def _e_step(self):
        temp_beta = np.zeros((self.K, self.V))
        for d, doc in enumerate(self.corpus):
            temp_gamma = np.zeros(self.K) + self.alpha
            for word in self.corpus[doc]:
                w = self.vocab.index(word)
                # Line (6) in Figure 6. 
                self._phi[doc][word] = self._beta[:, w] * \
                    np.exp(psi(self._gamma[d, :])).T
                # Line (7) in Figure 6. 
                self._phi[doc][word] /= self._phi[doc][word].sum()
                # Line (8) in Figure 6. 
                temp_gamma += self._phi[doc][word] * self.corpus[doc][word]
                temp_beta[:, w] += self._phi[doc][word] * self.corpus[doc][word]
            self._gamma[d, :] = temp_gamma
        print(self.get_theta()[0])
        return temp_beta


def main():
    # Reading the dataset.
    corpus_panda = pd.read_pickle(HOME_PATH + '/Projects/tminm/heavy_pickles/corpus.pickle')
    corpus = corpus_panda.to_dict()

    vocab_panda = pd.read_pickle(HOME_PATH + '/Projects/tminm/heavy_pickles/' + 'vocab.pickle')
    vocab = vocab_panda.tolist()


    # Running the LDA.
    v_lda = Variational_LDA(iter_count=25, K=10, alpha=1./10, eta=1./10)
    v_lda.fit(corpus=corpus, vocab=vocab)

    # Saving the results.
    theta = v_lda.get_theta()
    print(theta)
    theta.dump("../pickles/document_topic_probabilities_temp.pickle")


if __name__ == '__main__':
    main()
