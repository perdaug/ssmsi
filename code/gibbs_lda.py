import os
import pandas as pd
import numpy as np
from scipy.special import psi

HOME_PATH = os.path.expanduser('~')

class Gibbs_LDA(object):

    def __init__(self, corpus, K, alpha, eta):
        """
        Arguments:
        corpus: Collection of (mass, intensity) metabolite tuples;
        K: Number of topics;
        alpha: Hyperparameter for prior on weight vectors theta;
        eta: Hyperparameter for prior on topics beta.
        """
        self.vocab = self._read_vocab()
        exit()
        self.corpus = corpus
        self._K = K
        self._alpha = alpha
        self._eta = eta
        self._V = len(self.vocabulary)
        self._M = len(self.corpus)

    def _read_vocab(self):
        vocab_series = pd.read_pickle(HOME_PATH + '/Projects/tminm/heavy_pickles/vocab.pickle')
        vocab = vocab_series.tolist()
        return vocab

def main():
    # Reading the dataset.
    corpus_series = pd.read_pickle(HOME_PATH + '/Projects/tminm/heavy_pickles/corpus.pickle')
    corpus = corpus_series.to_dict()

    # Defining the number of topics and iterations.
    K = 10
    iterations = 100

    # Running the LDA.
    gibbs_lda = Gibbs_LDA(corpus, K, 1./K, 1./K)
    exit()
    gibbs_lda.run(iterations)

    # Saving the results.
    topic_probabilities = v_lda._gamma / v_lda._gamma.sum(axis=1)[:, None]
    topic_probabilities.dump("../pickles/document_topic_probabilities_temp.pickle")


if __name__ == '__main__':
    main()