import os
import numpy as np
import pandas as pd

from scipy.special import polygamma as pg
from scipy.special import psi as psi

HOME_PATH = os.path.expanduser('~')

class LDA(object):
    def __init__(self, corpus, topic_count):
        self.corpus = corpus

        self.k = topic_count
        self.doc_count = len(self.corpus)
        self.word_count = count_words(self.corpus)

        self.eta = 0.1
        self.alpha = 1
        self.init_beta()
        self.init_phi()
        self.gamma = np.zeros((self.doc_count, self.k))

    def init_beta(self):
        self.beta = np.random.rand(self.k, self.word_count)
        self.beta /= self.beta.sum(axis=1)[:,None]

    def init_phi(self):
        self.phi = {}
        for doc in self.corpus:
            self.phi[doc] = {}
            for word in self.corpus[doc]:
                self.phi[doc][word] = np.zeros(self.k) 

    def run(self, iteration_count):
        # Initialisisation.
        for i, doc in enumerate(self.corpus):
            N = 0
            for word in self.corpus[doc]:
                N += self.corpus[doc][word]
            self.gamma[i, :] = self.alpha + N / self.k

        # The main loop.
        for i in range(iteration_count):
            temp_beta = np.zeros((self.k, self.word_count))
            # e step
            for doc_id, doc in enumerate(self.corpus):
                temp_gamma = np.zeros(self.k) + self.alpha
                for word_id, word in enumerate(self.corpus[doc]):
                    self.phi[doc][word] = self.beta[:, word_id] * np.exp(psi(self.gamma[doc_id, :])).T
                    self.phi[doc][word] /= self.phi[doc][word].sum()
                    temp_gamma += self.phi[doc][word] * self.corpus[doc][word]
                    temp_beta[:, word_id] += self.phi[doc][word] * self.corpus[doc][word]
                self.gamma[doc_id, :] = temp_gamma
            temp_beta += self.eta
            temp_beta /= temp_beta.sum(axis=1)[:,None]
            self.beta = temp_beta
            self.alpha = self.update_alpha()

    def update_alpha(self, iteration_count=20, init_alpha=[]):
        M,K = self.gamma.shape
        if not len(init_alpha) > 0:
            init_alpha = self.gamma.mean(axis=0) / K
        alpha = init_alpha.copy()
        g = (psi(self.gamma) - psi(self.gamma.sum(axis=1))[:,None]).sum(axis=0)
        for i in range(iteration_count):
            grad = M * (psi(alpha.sum()) - psi(alpha)) + g
            H = -M * np.diag(pg(1, alpha)) + M * pg(1, alpha.sum())
            alpha_new = alpha - np.dot(np.linalg.inv(H), grad)
            if (alpha_new < 0).sum() > 0:
                init_alpha /= 10.0
                return self.update_alpha(iteration_count=iteration_count, init_alpha = init_alpha)
            diff = np.sum(np.abs(alpha - alpha_new))
            alpha = alpha_new
            if diff < 1e-6 and i > 1:
                return alpha
        return alpha



def main():
    corpus_series = pd.read_pickle('../heavy_pickles/corpus.pickle')
    corpus = corpus_series.to_dict()

    lda_in_dev = LDA(corpus, 10)
    lda_in_dev.run(10)
    print lda_in_dev.gamma

def count_words(corpus):
    words = []
    for doc in corpus:
        for word in corpus[doc]:
            if word not in words:
                words.append(word)
    return len(words)



if __name__ == '__main__':
    main()
