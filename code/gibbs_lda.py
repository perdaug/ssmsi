import os
import logging
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
        self.iter_count = iter_count
        self._K = K
        self._alpha = alpha
        self._eta = eta

    def fit(self, corpus):
        self._fit(corpus)

    def dump_thetas(self, path):
        normalised_thetas = self._theta / self._sampling_count
        normalised_thetas.dump(path)

    def dump_phis(self, path):
        normalised_phis = self._phi / self._sampling_count
        normalised_phis.dump(path)

    def _fit(self, corpus):
        self._initialise(corpus)

        thetas_over_time = []

        for it in range(self.iter_count):
            print("Iteration %s." % it) 
            self._sample_topics(corpus)

            self._sampling_count += 1.0
            for d, _ in enumerate(corpus):
                z_counts = self._dz_counts[d, :]
                self._theta[d, :] += np.random.dirichlet(z_counts)
            for z in range(self._K):
                w_counts = self._zw_counts[z, :]
                self._phi[z, :] += np.random.dirichlet(w_counts)
            current_theta = self._theta / self._sampling_count
            print("Current theta:\n %s." % current_theta) 

            thetas_over_time.append(current_theta)

        pickle = np.array(thetas_over_time)
        pickle.dump(HOME_PATH + '/Projects/tminm/heavy_pickles/thetas_over_time.pickle')


    def _initialise(self, corpus):
        self._D, self._W = corpus.shape
        self._total_frequencies = corpus.sum()

        self._zw_counts = np.zeros((self._K, self._W), dtype=np.int)
        self._dz_counts = np.zeros((self._D, self._K), dtype=np.int)
        self._z_counts = np.zeros(self._K, dtype=np.int)
        self._z_history = []

        for d, doc in enumerate(corpus):
            self._z_history.append([])
            for w, word in enumerate(doc):
                self._z_history[d].append([])
                word_frequency = int(word)
                for _ in range(word_frequency):
                    z = np.random.randint(self._K)
                    self._z_history[d][w].append(z)
                    self._dz_counts[d, z] += 1
                    self._zw_counts[z, w] += 1
                    self._z_counts[z] += 1

        self._sampling_count = 0.0
        self._theta = np.zeros((self._D, self._K), np.float)
        self._phi = np.zeros((self._K, self._W), np.float)


    def _sample_topics(self, corpus):
        for d, doc in enumerate(corpus):
            for w, _ in enumerate(doc):
                for it, z in enumerate(self._z_history[d][w]):
                    self._dz_counts[d, z] -= 1
                    self._zw_counts[z, w] -= 1
                    self._z_counts[z] -= 1

                    p_topic = (self._zw_counts[:, w] + self._eta) / \
                        (self._z_counts + self._eta * self._K) * \
                        (self._dz_counts[d, :] + self._alpha)
                    p_topic = p_topic / p_topic.sum()
                    z_new = np.random.choice(self._K, p=p_topic)

                    self._z_history[d][w][it] = z_new
                    self._dz_counts[d, z_new] += 1
                    self._zw_counts[z_new, w] += 1
                    self._z_counts[z_new] += 1

def main():
    K = 10
    corpus = pd.read_pickle(HOME_PATH + '/Projects/tminm/heavy_pickles/preprocessed_corpus.pickle')

    gibbs_lda = Gibbs_LDA(K, iter_count=50, alpha=1./K, eta=1./K)
    gibbs_lda.fit(corpus)
    gibbs_lda.dump_thetas(HOME_PATH + '/Projects/tminm/pickles/' + 'document_topic_probabilities_ari_gibbs.pickle')


if __name__ == '__main__':
    main()
