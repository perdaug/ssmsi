import os
import pandas as pd
import numpy as np

HOME_PATH = os.path.expanduser('~')
np.set_printoptions(edgeitems=5)


class Gibbs_LDA(object):

    def __init__(self, K, iter_count):
        self.iter_count = iter_count
        self.K = K
        self.thetas_ = []

    def fit(self, corpus):
        self._fit(corpus)

    def dump_thetas(self, path):
        pickle = np.array(self.thetas_)
        pickle.dump(path)

    def _fit(self, corpus):
        self._initialise(corpus)

        for it in range(self.iter_count):
            print("Iteration %s:" % it) 
            self._sample_topics(corpus)
            
            self._store_theta()
            print("Current theta:")
            print(self.thetas_[it]) 

    def _initialise(self, corpus):
        self.D, self.W = corpus.shape
        self._total_frequencies = corpus.sum()
        self.alpha = 1. / self.K
        self.beta = 1. / self.W

        self._zw_counts = np.zeros((self.K, self.W), dtype=np.int)
        self._dz_counts = np.zeros((self.D, self.K), dtype=np.int)
        self._z_counts = np.zeros(self.K, dtype=np.int)
        self._z_history = []

        for d, doc in enumerate(corpus):
            self._z_history.append([])
            for w, word in enumerate(doc):
                self._z_history[d].append([])
                word_frequency = int(word)
                for _ in range(word_frequency):
                    z = np.random.randint(self.K)
                    self._z_history[d][w].append(z)
                    self._dz_counts[d, z] += 1
                    self._zw_counts[z, w] += 1
                    self._z_counts[z] += 1

    def _sample_topics(self, corpus):
        for d, doc in enumerate(corpus):
            for w, _ in enumerate(doc):
                for it, z in enumerate(self._z_history[d][w]):
                    self._dz_counts[d, z] -= 1
                    self._zw_counts[z, w] -= 1
                    self._z_counts[z] -= 1

                    # The second denominator from eq. 5 (Griffiths and Stevyers 2004) is omitted because it is a constant and we normalise the vector p_topic anyway.
                    p_topic = ((1.0 * self._zw_counts[:, w] + self.beta) / \
                        (1.0 * self._z_counts + self.W * self.beta)) * \
                        ((self._dz_counts[d, :] + self.alpha))
                    p_topic = p_topic / p_topic.sum()
                    z_new = np.random.choice(self.K, p=p_topic)

                    self._z_history[d][w][it] = z_new
                    self._dz_counts[d, z_new] += 1
                    self._zw_counts[z_new, w] += 1
                    self._z_counts[z_new] += 1

    def _store_theta(self):
        theta = self._dz_counts + self.alpha
        theta /= np.sum(theta, axis=1)[:, np.newaxis]
        theta = np.random.dirichlet(theta)
        self.thetas_.append(theta)


def main():
    corpus = pd.read_pickle(HOME_PATH + '/Projects/tminm/heavy_pickles/preprocessed_corpus.pickle')

    gibbs_lda = Gibbs_LDA(K=10, iter_count=50)
    gibbs_lda.fit(corpus)
    gibbs_lda.dump_thetas(HOME_PATH + '/Projects/tminm/pickles/' + 'document_topic_probabilities_ari_gibbs.pickle')


if __name__ == '__main__':
    main()
