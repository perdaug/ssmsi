import os
import pandas as pd
import numpy as np

HOME_PATH = os.path.expanduser('~')
np.set_printoptions(edgeitems=5)


class Dynamic_Gibbs_LDA(object):

    def __init__(self, K, iter_count, n_seq):
        self.iter_count = iter_count
        self.K = K
        self.n_seq = n_seq
        self.thetas_ = []
        self.phis_ = []
        self.alphas_ = []

    def fit(self, corpus):
        self._fit(corpus)

    def dump(self, array, path):
        pickle = np.array(array)
        pickle.dump(path)

    def _fit(self, corpus):
        list_of_seq_corpus = np.array_split(corpus, self.n_seq)

        for seq_c, seq_corpus in enumerate(list_of_seq_corpus):

            self._initialise(seq_corpus, seq_c)
            print(self.alpha.shape)
            print(self.alpha)
            print(self.alpha.sum())
            print(self.beta.shape)
            print(self.beta[0, :])
            print(self.beta[0, :].sum())

            for it in range(self.iter_count):
                
                print("Sequence: %s, Iteration: %s." % (seq_c, it)) 
                self._sample_topics(seq_corpus)
                self._store_theta()
                self._store_phi()
                print("Current theta:")
                print(self.thetas_[it]) 

            self._dynamic_step(seq_c)


    def _initialise(self, corpus, iteration):
        self.D, self.W = corpus.shape
        self._total_frequencies = corpus.sum()
        if iteration == 0:
            self.alpha = np.full(self.K, 1./self.K)
            self.beta = np.full((self.K, self.W), 1./self.W)

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
                    p_topic = ((1.0 * self._zw_counts[:, w] + self.beta[:, w]) / \
                        (1.0 * self._z_counts + self.W * self.beta[:, w])) * \
                        ((self._dz_counts[d, :] + self.alpha))
                    p_topic = p_topic / p_topic.sum()
                    z_new = np.random.choice(self.K, p=p_topic)

                    self._z_history[d][w][it] = z_new
                    self._dz_counts[d, z_new] += 1
                    self._zw_counts[z_new, w] += 1
                    self._z_counts[z_new] += 1

    def _dynamic_step(self, iteration):
        # fix index
        index = iteration * self.iter_count + self.iter_count - 1
        self.alpha = np.sum(self.thetas_[index], axis=0)
        self.alpha /= self.alpha.sum()
        self.alphas_.append(self.alpha)
        self.beta = self.phis_[index]

    def _store_theta(self):
        theta = self._dz_counts + self.alpha
        theta /= np.sum(theta, axis=1)[:, np.newaxis]
        # theta = np.random.dirichlet(theta)
        self.thetas_.append(theta)

    def _store_phi(self):
        phi = self._zw_counts + self.beta
        phi /= np.sum(phi, axis=1)[:, np.newaxis]
        # phi = np.random.dirichlet(phi)
        self.phis_.append(phi)


def main():
    corpus = pd.read_pickle(HOME_PATH + '/Projects/tminm/heavy_pickles/preprocessed_corpus.pickle')

    d_gibbs_lda = Dynamic_Gibbs_LDA(K=10, iter_count=1, n_seq=1)
    d_gibbs_lda.fit(corpus)
    d_gibbs_lda.dump(d_gibbs_lda.thetas_, HOME_PATH + '/Projects/tminm/pickles/' + 'document_topic_probabilities_ari_gibbs.pickle')
    d_gibbs_lda.dump(d_gibbs_lda.alphas_, HOME_PATH + '/Projects/tminm/pickles/' + 'document_alphas_ari_gibbs.pickle')


if __name__ == '__main__':
    main()
