# Implementation of a variational LDA.
# 
# Code refers to the variational LDA implementation of Dr Simon Rogers.
# Comments refer to the LDA paper (Blei et al. 2003)  
# and the the online LDA implementation (Hoffman et al. 2010).

import os
import pandas as pd
import numpy as np
from scipy.special import psi

np.set_printoptions(suppress=True)
HOME_PATH = os.path.expanduser('~')

class Variational_LDA(object):

    def __init__(self, corpus, K, alpha, eta):
        """
        Arguments:
        corpus: Collection of (mass, intensity) metabolite tuples;
        K: Number of topics;
        alpha: Hyperparameter for prior on weight vectors theta;
        eta: Hyperparameter for prior on topics beta.
        """
        print corpus.keys()
        exit()
        self.corpus = corpus
        self.vocabulary = create_vocabulary(corpus)
        self._K = K
        self._alpha = alpha
        self._eta = eta
        self._V = len(self.vocabulary)
        self._M = len(self.corpus)

    def init_vb(self):
        # Initialising Beta.
        self._beta = np.random.rand(self._K, self._V) 
        self._beta /= self._beta.sum(axis=1)[:, None]

        # Initialising Gamma and Phi.
        self._phi = {}
        self._gamma = np.zeros((self._M, self._K))
        for d, doc in enumerate(self.corpus):
            self._phi[doc] = {}
            for word in self.corpus[doc]:
                w = self.vocabulary[word]
                # Line (1) in Figure 6.
                self._phi[doc][w] = np.zeros(self._K)
            total_intensity = sum(self.corpus[doc].values())
            # Line (2) in Figure 6 .
            self._gamma[d, :] = self._alpha + total_intensity / self._K

    def run(self, iterations):
        self.init_vb()

        for _ in range(iterations):
            self.vb_step()

    def vb_step(self):
        temp_beta = self.e_step()
        temp_beta += self._eta
        temp_beta /= temp_beta.sum(axis=1)[:, None]
        self._beta = temp_beta

    def e_step(self):
        temp_beta = np.zeros((self._K, self._V))
        for d, doc in enumerate(self.corpus):
            temp_gamma = np.zeros(self._K) + self._alpha
            for word in self.corpus[doc]:
                w = self.vocabulary[word]
                # Line (6) in Figure 6. 
                self._phi[doc][w] = self._beta[:, w] * \
                    np.exp(psi(self._gamma[d, :])).T
                # Line (7) in Figure 6. 
                self._phi[doc][w] /= self._phi[doc][w].sum()
                # Line (8) in Figure 6. 
                temp_gamma += self._phi[doc][w] * self.corpus[doc][word]
                temp_beta[:, w] += self._phi[doc][w] * self.corpus[doc][word]
            self._gamma[d, :] = temp_gamma
        return temp_beta

def main():
    # Reading the dataset.
    corpus_series = pd.read_pickle(HOME_PATH + '/Projects/mlinb/heavy_pickles/corpus.pickle')
    corpus = corpus_series.to_dict()

    # Defining the number of topics and iterations.
    K = 10
    iterations = 100

    # Running the LDA.
    v_lda = Variational_LDA(corpus, K, 1./K, 1./K)
    v_lda.run(iterations)

    # Saving the results.
    topic_probabilities = v_lda._gamma / v_lda._gamma.sum(axis=1)[:, None]
    topic_probabilities.dump("../pickles/document_topic_probabilities_temp.pickle")


if __name__ == '__main__':
    main()
