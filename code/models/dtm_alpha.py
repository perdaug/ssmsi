
"""
VERSION
- Python 2

FUNCTION
- Infer topics by updating the alphas.
"""

import os
import pandas as pd
import numpy as np
import copy
from time import time

np.set_printoptions(threshold=np.nan)

# ___________________________________________________________________________


class DTM_Alpha(object):
    '''
    PARAMETERS:
    - sigma_0_sq is the initial variance for alpha[0];
    - sigma_sq is the variance for alpha[t]
    - delta_sq is the variance for alpha'[t]
    '''
    def __init__(self, K=None, beta=None, sigma_0_sq=None, sigma_sq=None,
                 delta_sq=None, autoreg=False, snapshot=False,
                 verbose_init=True):
        self.K = K
        self.sigma_0_sq = sigma_0_sq
        self.sigma_sq = sigma_sq
        self.delta_sq = delta_sq
        self.autoreg = autoreg
        self.snapshot = snapshot
        self.verbose_init = verbose_init
        self.beta = beta
        '''
        Counter and array initialisation
        '''
        self.n_it_sampl_last = None
        self.n_upd_alpha = None
        self.n_it_alpha = None
        self.hist_theta = None
        self.hist_alpha = None
        self.hist_phi = None
        '''
        The parameters initialised later in the work-flow
        '''
        self.burnin = None
        self.T = None
        self.V = None
        self.alpha = None
        self.n_zw = None
        self.n_dz = None
        self.n_z = None
        self.hist_z = None
        self.corpus = None
# ===========================================================================

    def load_fit(self, path_file, n_it_add=0):
        vars_loaded = pd.read_pickle(path_file)
        for var in vars(self):
            if var is 'snapshot':
                setattr(self, 'snapshot', True)
            else:
                setattr(self, var, vars_loaded[var])
        self.fit(n_it=n_it_add)
# ===========================================================================

    def fit(self, n_it, n_burn_it=0, corpus=None):
        print('{} has started.'.format(self.__class__.__name__))
        if not self.snapshot:
            self._initialise(corpus)
        if self.burnin:
            print('The burn-in started.')
        for it in range(self.n_it_sampl_last, self.n_it_sampl_last + n_it):
            self._update_status(it, n_burn_it)
            self._update_alpha()
            self._assign_topics()
            if not self.burnin:
                self._sample_phi()
                self._sample_theta()
                # self._calculate_perplexity()
            if self.verbose_init:
                self.verbose_init = False
        self.n_it_sampl_last += n_it
        print('{} has finished.'.format(self.__class__.__name__))

    def _update_status(self, it, n_burn_it):
        if it != 0 and it % 5 == 0:
            print('Iteration: {}'.format(it))
            rate_alpha_update = 1.0 * self.n_upd_alpha / self.n_it_alpha
            print('Alpha update rate: {:.2f}'.format(rate_alpha_update))
        if it == n_burn_it:
            print('Burn-in finished.')
            self.burnin = False
# ___________________________________________________________________________

    def _initialise(self, corpus):
        t0 = time()
        '''
        Initialise auxiliary variables
        '''
        self.burnin = True
        self.n_it_sampl_last = 0
        self.n_upd_alpha = 0
        self.n_it_alpha = 0
        self.hist_theta = []
        self.hist_alpha = []
        self.hist_phi = []
        self.corpus = corpus
        self.T, self.V = corpus.shape
        '''
        Initialise hyper-parameters
        '''
        self.alpha = np.zeros(shape=(self.T, self.K))
        self.alpha[0] = np.random.normal(0, self.sigma_0_sq, self.K)
        for t in range(1, self.T):
            if self.autoreg:
                self.alpha[t] = np.random.normal(self.alpha[t - 1],
                                                 self.sigma_sq)
            else:
                self.alpha[t] = np.random.normal(0, self.sigma_0_sq)
        '''
        Run the initial assignment.
        '''
        self.n_zw = np.zeros((self.K, self.V), dtype=np.int)
        self.n_dz = np.zeros((self.T, self.K), dtype=np.int)
        self.n_z = np.zeros(self.K, dtype=np.int)
        self.hist_z = []
        for d, doc in enumerate(corpus):
            self.hist_z.append([])
            for w, word in enumerate(doc):
                self.hist_z[d].append([])
                n_word = int(word)
                for _ in range(n_word):
                    z = np.random.randint(self.K)
                    self.hist_z[d][w].append(z)
                    self.n_dz[d, z] += 1
                    self.n_zw[z, w] += 1
                    self.n_z[z] += 1
        if self.verbose_init:
            print('Initialisation run-time: {:.2f}'.format(time() - t0))
# ___________________________________________________________________________

    '''
    TERMS:
    - p: Probability
    - t: Term
    - ar: Autoregressive
    - r: Acceptance rate

    FORMULA:
    - p(z_k,a_k|x)=p(x|z_k,a_k)*p(z_k|a_k)*p(a_k):
    -- t1: Mult(softmax(a)_k)
    -- t3: p(a_{k,t}|a_{k,t-1})*p(a_{k,t+1}|a_{k,t})
    --- t31: p(a_{k,t}|a_{k,t-1})
    --- t32: p(a_{k,t+1}|a_{k,t})
    '''
    def _update_alpha(self):
        t0 = time()
        self.hist_alpha.append(copy.deepcopy(self.alpha))
        for t in range(0, self.T):
            alpha_prop = np.random.normal(self.alpha[t], self.delta_sq)
            alpha_updated = copy.deepcopy(self.alpha[t])
            for k in range(0, self.K):
                '''
                Calculating p(z_k,a_k,a'_k|x)
                '''
                theta_prop = self._softmax_log_vector(alpha_prop)
                t1_prop = np.sum(self.n_dz[t] * theta_prop)
                if self.autoreg:
                    t3_prop = self._eval_prior_ar(alpha_prop, t, k)
                else:
                    t3_prop = self._eval_norm(alpha_prop[k], 0,
                                              self.sigma_0_sq)
                p_alpha_prop = t1_prop + t3_prop
                '''
                Calculating p(z_k,a_k|x)
                '''
                theta = self._softmax_log_vector(self.alpha[t])
                t1 = np.sum(self.n_dz[t] * theta)
                if self.autoreg:
                    t3 = self._eval_prior_ar(self.alpha[t], t, k)
                else:
                    t3 = self._eval_norm(self.alpha[t][k], 0,
                                         self.sigma_0_sq)
                p_alpha = t1 + t3
                '''
                Calculating the acceptance rate
                '''
                r = p_alpha_prop - p_alpha
                r_norm = np.exp(np.minimum(0, r))
                accept_alpha = np.random.binomial(1, r_norm)
                if accept_alpha:
                    alpha_updated[k] = alpha_prop[k]
                    self.n_upd_alpha += 1
                self.n_it_alpha += 1
            self.alpha[t] = alpha_updated
        if self.verbose_init:
            print('Alpha update run-time: {:.2f}'.format(time() - t0))

    '''
    - The constant is taken away;
    - Executed in the log space.
    '''
    def _eval_norm(self, x, mean, dev_sq):
        product = np.dot(np.transpose(x - mean), (x - mean))
        return -0.5 * product / dev_sq

    def _softmax_log(self, arr, i):
        softmax_linear = np.exp(arr[i]) / np.sum(np.exp(arr))
        return np.log(softmax_linear)

    def _softmax_log_vector(self, arr):
        softmax = np.exp(arr) / np.sum(np.exp(arr))
        return np.log(softmax)

    def _eval_prior_ar(self, alpha_prop, t, k):
        if t == 0:
            t31 = self._eval_norm(alpha_prop[k], 0, self.sigma_0_sq)
        else:
            t31 = self._eval_norm(alpha_prop[k], self.alpha[t - 1][k],
                                  self.sigma_sq)
        if t == self.T - 1:
            t3 = t31
        else:
            t32 = self._eval_norm(self.alpha[t + 1][k], alpha_prop[k],
                                  self.sigma_sq)
            t3 = t31 + t32
        return t3
# ___________________________________________________________________________

    def _assign_topics(self):
        t0 = time()
        self.alpha_softmax = self._softmax(self.alpha)
        for d, doc in enumerate(self.corpus):
            for w, _ in enumerate(doc):
                for it, z in enumerate(self.hist_z[d][w]):
                    self.n_dz[d, z] -= 1
                    self.n_zw[z, w] -= 1
                    self.n_z[z] -= 1
                    p_topic = self._calculate_p_topic(d, w)
                    z_new = np.random.choice(self.K, p=p_topic)
                    self.hist_z[d][w][it] = z_new
                    self.n_dz[d, z_new] += 1
                    self.n_zw[z_new, w] += 1
                    self.n_z[z_new] += 1
        if self.verbose_init:
            print('Gibbs sampling run-time: {:.2f}'.format(time() - t0))

    def _calculate_p_topic(self, idx_d, idx_w):
        term_left = (1.0 * self.n_zw[:, idx_w] + self.beta) \
            / (1.0 * self.n_z + self.V * self.beta)
        term_right = self.alpha_softmax[idx_d]
        p_topic = term_left * term_right
        p_topic /= p_topic.sum()
        return p_topic

    def _softmax(self, arr):
        arr_softmax = np.array(arr)
        for idx_r, row in enumerate(arr):
            arr_softmax[idx_r] = np.exp(row) / np.sum(np.exp(row))
        return arr_softmax
# ___________________________________________________________________________

    def _sample_phi(self):
        alphas_dir = (1.0 * self.n_zw + self.beta)
        phi = []
        for alpha_dir in alphas_dir:
            phi_topic = np.random.dirichlet(alpha_dir)
            phi.append(phi_topic)
        self.hist_phi.append(phi)
# ___________________________________________________________________________

    def _sample_theta(self):
        alphas_dir = self.alpha_softmax
        theta = []
        for alpha_dir in alphas_dir:
            theta_doc = np.random.dirichlet(alpha_dir)
            theta.append(theta_doc)
        self.hist_theta.append(np.array(theta))

# ___________________________________________________________________________
# BACKLOG

    def _calculate_perplexity(self):
        perp = 0
        for t in range(self.T):
            for v in range(self.V):
                sum_k = 0
                for k in range(self.K):
                    sum_k += self.hist_theta[-1][t, k] * self.beta[k, v]
                perp += self.corpus[t, v] * np.log(sum_k)
        perp /= self.corpus.sum()
        perp = np.exp(-perp)

# ___________________________________________________________________________

PATH_HOME = os.path.expanduser('~') + '/Projects/ssmsi/'
PATH_DATA = PATH_HOME + 'data/corpora_processed/'


def main():
    corpus_pp = pd.read_pickle(PATH_DATA + 'corpus_synthetic_nparray.pkl')
    # vocab = pd.read_pickle(PATH_DATA + 'vocab_synthetic.pkl')
    # sigma_0_sq = 1
    # sigma_sq = 0.01
    # delta_sq = 1
    # K = 2
    # autoreg = True
    # clf = DTM_Alpha(K=K, sigma_0_sq=sigma_0_sq, sigma_sq=sigma_sq,
    #                 delta_sq=delta_sq, autoreg=autoreg)
    # beta = np.array([[0, 0, 0, 0.3, 0.7, 0, 0, 0, 0, 0],
    #                 [0, 0, 0.2, 0, 0, 0, 0, 0.3, 0.3, 0.2]])
    # clf.beta = beta
    # n_it = 10
    # clf.fit(n_it=n_it, corpus=corpus_pp)


if __name__ == '__main__':
    main()
