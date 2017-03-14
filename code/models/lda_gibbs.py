
import os
import pandas as pd
import numpy as np
from time import time


# ___________________________________________________________________________


class LDA_Gibbs(object):

    def __init__(self, K, snapshot=False, verbose_init=True):
        self.n_it_sampl_last = 0
        self.K = K
        self.hist_thetas = []
        self.hist_phis = []
        self.snapshot = snapshot
        self.corpus = None
        self.verbose_init = verbose_init

# ___________________________________________________________________________
    def fit(self, n_it, corpus=None):
        if not self.snapshot:
            self._initialise(corpus)
        for it in range(self.n_it_sampl_last, self.n_it_sampl_last + n_it):
            if (it % 5 == 0):
                print("Iteration %s:" % it)
            self._sample_topics()
            self._store_theta()
            self._store_phi()
            if self.verbose_init:
                self.verbose_init = False
        self.n_it_sampl_last += n_it
# ___________________________________________________________________________

    def load_fit(self, path_file, n_it=0):
        vars_loaded = pd.read_pickle(path_file)
        for var in vars(self):
            if var is 'snapshot':
                setattr(self, 'snapshot', True)
            else:
                setattr(self, var, vars_loaded[var])
        self.fit(n_it=n_it)
# ___________________________________________________________________________

    def _initialise(self, corpus):
        t0 = time()
        print('{} has started.'.format(self.__class__.__name__))
        self.corpus = corpus
        self.D, self.W = self.corpus.shape
        '''
        Initialise hyper-parameters
        '''
        self.alpha = 1. / self.K
        self.beta = 1. / self.W
        '''
        Run the initial assignment.
        '''
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
        if self.verbose_init:
            print('Initialisation run-time: {:.2f}'.format(time() - t0))

# ___________________________________________________________________________

    def _sample_topics(self):
        t0 = time()
        for d, doc in enumerate(self.corpus):
            # t0 = time()
            for w, _ in enumerate(doc):
                for it, z in enumerate(self._z_history[d][w]):
                    self._dz_counts[d, z] -= 1
                    self._zw_counts[z, w] -= 1
                    self._z_counts[z] -= 1
                    theta = (self._dz_counts[d, :] + self.alpha)
                    phi = ((1.0 * self._zw_counts[:, w] + self.beta) /
                           (1.0 * self._z_counts + self.W * self.beta))
                    p_topic = phi * theta
                    p_topic /= p_topic.sum()
                    z_new = np.random.choice(self.K, p=p_topic)
                    self._z_history[d][w][it] = z_new
                    self._dz_counts[d, z_new] += 1
                    self._zw_counts[z_new, w] += 1
                    self._z_counts[z_new] += 1
            # print('Gibbs sampling run-time: {:.2f}'.format(time() - t0))
            
        if self.verbose_init:
            print('Gibbs sampling run-time: {:.2f}'.format(time() - t0))

# ___________________________________________________________________________

    def _store_theta(self):
        theta_pre = (self._dz_counts + self.alpha)
        theta = self.softmax(theta_pre)
        self.hist_thetas.append(np.array(theta))

    def softmax(self, arr):
        for idx_r, row in enumerate(arr):
            arr[idx_r] = np.exp(row) / np.sum(np.exp(row))
        return arr
# ___________________________________________________________________________

    def _store_phi(self):
        phi_pre = ((1.0 * self._zw_counts + self.beta) /
                   (1.0 * self._z_counts[:, None] + self.W * self.beta))
        phi = self.softmax(phi_pre)
        self.hist_phis.append(np.array(phi))
        # hist_thetas_p_norm = phi / np.sum(phi, axis=1)[:, None]

        # hist_thetas_current = np.zeros(shape=phi.shape)
        # for doc_idx, theta_p_n in enumerate(hist_thetas_p_norm):
        #     theta_current = np.random.dirichlet(theta_p_n)
        #     hist_thetas_current[doc_idx, :] = theta_current
# ___________________________________________________________________________


PATH_HOME = os.path.expanduser('~') + '/Projects/ssmsi/'
PATH_DATA = PATH_HOME + 'data/corpora_processed/'


def main():
    lda_gibbs = LDA_Gibbs(K=10, iter_count=opts.iter)
    corpus = pd.read_pickle(PATH_DATA + 'vocab_bne.pkl')
    lda_gibbs.fit(corpus)


if __name__ == '__main__':
    main()
