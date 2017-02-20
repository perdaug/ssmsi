import os
import pandas as pd
import numpy as np
import copy
from scipy.stats import norm
import pickle as pkl

np.set_printoptions(threshold=np.nan)


class DTM_Alpha(object):

    '''
    Parameters:
    - sigma_0_sq is the initial variance for alpha[0];
    - sigma_sq is the variance for alpha[t]
    - delta_sq is the variance for alpha'[t]
    '''
    def __init__(self, K=None, sigma_0_sq=None, sigma_sq=None, delta_sq=None,
                 autoreg=False, snapshot=False):
        # Input
        self.K = K
        self.sigma_0_sq = sigma_0_sq
        self.sigma_sq = sigma_sq
        self.delta_sq = delta_sq
        self.autoreg = autoreg
        # Counters and array initialisation
        self.n_it_sampl_last = 0
        self.n_upd_alpha = 0
        self.n_it_alpha = 0
        self.hist_theta = []
        self.hist_alpha = []
        self.hist_z = []
        # Parameters initialised later in the work-flow
        self.snapshot = snapshot
        self.beta = None
        self.T = None
        self.V = None
        self.alpha = None
        self.n_zw = None
        self.n_dz = None
        self.n_z = None
        self.corpus = None
# ___________________________________________________________________________

    def fit(self, n_it, corpus=None):
        if not self.snapshot:
            self._initialise(corpus)
        for it in range(self.n_it_sampl_last, self.n_it_sampl_last + n_it):
            if it != 0 and it % 10 == 0:
                self._print_update(it)
            self.hist_alpha.append(copy.deepcopy(self.alpha))
            self._update_alpha()
            self._sample_topics()
            self._store_theta()
        self.n_it_sampl_last += n_it

    def _print_update(self, it):
        print 'Iteration: %s' % self.n_upd_alpha
        print 'Total alpha changes: %s' % self.n_upd_alpha
        print 'Alpha update rate: %.4f' \
            % (1.0 * self.n_upd_alpha / self.n_it_alpha)
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
        # Initialise the parameters
        # self.n_it = n_it
        self.corpus = corpus
        self.T, self.V = corpus.shape
        self.alpha = np.zeros(shape=(self.T, self.K))
        self.alpha[0] = np.random.normal(0, self.sigma_0_sq, self.K)
        for t in xrange(1, self.T):
            if self.autoreg:
                self.alpha[t] = np.random.normal(self.alpha[t-1],
                                                 self.sigma_sq)
            else:
                self.alpha[t] = np.random.normal(0, self.sigma_0_sq)
        self.n_zw = np.zeros((self.K, self.V), dtype=np.int)
        self.n_dz = np.zeros((self.T, self.K), dtype=np.int)
        self.n_z = np.zeros(self.K, dtype=np.int)

        # Run the initial assignments
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
# ___________________________________________________________________________

    '''
    Syntax:
    - p: Probability
    - t: Term
    - ar: Autoregressive
    - r: Acceptance rate

    Formulae:
    - p(z_k,a_k|x)=p(x|z_k,a_k)*p(z_k|a_k)*p(a_k):
    -- t1: Mult(softmax(a)_k)
    -- t2: softmax(a)_k
    -- t3: p(a_{k,t}|a_{k,t-1})*p(a_{k,t+1}|a_{k,t})
    --- t31: p(a_{k,t}|a_{k,t-1})
    --- t32: p(a_{k,t+1}|a_{k,t})
    '''
    def _update_alpha(self):
        for t in xrange(0, self.T):
            alpha_prop = np.random.normal(self.alpha[t], self.delta_sq)
            alpha_updated = copy.deepcopy(self.alpha[t])
            for k in xrange(0, self.K):
                # Calculating p(z_k,a_k,a'_k|x)
                theta_prop = self._softmax_log(alpha_prop, k)
                t1_prop = self.n_dz[t][k] * theta_prop
                t2_prop = theta_prop
                if self.autoreg:
                    t3_prop = self._eval_prior_ar(alpha_prop, t, k)
                else:
                    t3_prop = self._eval_norm(alpha_prop[k], 0,
                                              self.sigma_0_sq)
                p_alpha_prop = t1_prop + t2_prop + t3_prop
                # Calculating p(z_k,a_k|x)
                theta = self._softmax_log(self.alpha[t], k)
                t1 = self.n_dz[t][k] * theta
                t2 = theta
                if self.autoreg:
                    t3 = self._eval_prior_ar(self.alpha[t], t, k)
                else:
                    t3 = self._eval_norm(self.alpha[t][k], 0,
                                         self.sigma_0_sq)
                p_alpha = t1 + t2 + t3
                # Calculating the acceptance rate
                r = np.exp(p_alpha_prop - p_alpha)
                r_norm = np.minimum(1, r)
                accept_alpha = np.random.binomial(1, r_norm)
                if accept_alpha:
                    alpha_updated[k] = alpha_prop[k]
                    self.n_upd_alpha += 1
                self.n_it_alpha += 1
            self.alpha[t] = alpha_updated

    '''
    - The constant is taken away;
    - Executed in the log space.
    '''
    def _eval_norm(self, x, mean, dev_sq):
        product = np.dot(np.transpose(x-mean), (x-mean))
        return -1./2 * product / dev_sq

    def _softmax_log(self, arr, i):
        softmax_linear = np.exp(arr[i]) / np.sum(np.exp(arr))
        return np.log(softmax_linear)

    def _eval_prior_ar(self, alpha_prop, t, k):
        if t == 0:
            t31 = self._eval_norm(alpha_prop[k], 0, self.sigma_0_sq)
        else:
            t31 = self._eval_norm(alpha_prop[k], self.alpha[t-1][k],
                                  self.sigma_sq)
        if t == self.T - 1:
            t3 = t31
        else:
            t32 = self._eval_norm(self.alpha[t+1][k], alpha_prop[k],
                                  self.sigma_sq)
            t3 = t31 + t32
        return t3
# ___________________________________________________________________________

    def _sample_topics(self):
        for d, doc in enumerate(self.corpus):
            for w, _ in enumerate(doc):
                for it, z in enumerate(self.hist_z[d][w]):
                    self.n_dz[d, z] -= 1
                    self.n_zw[z, w] -= 1
                    self.n_z[z] -= 1

                    # Eq. 5, the second denominator is  a constant
                    p_topic = self._calculate_p_topic(d, w)
                    z_new = np.random.choice(self.K, p=p_topic)
                    self.hist_z[d][w][it] = z_new
                    self.n_dz[d, z_new] += 1
                    self.n_zw[z_new, w] += 1
                    self.n_z[z_new] += 1

    def _softmax_vector(self, arr):
        return np.exp(arr) / np.sum(np.exp(arr))

    def _calculate_p_topic(self, idx_d, idx_w):
        # TODO: Look into the formula
        phi = (1.0 * self.n_zw[:, idx_w] + self.beta[:, idx_w]) \
            / (1.0 * self.n_z + self.V * self.beta[:, idx_w])
        theta = self.n_dz[idx_d, :] + self.alpha[idx_d]
        p_topic = theta * phi
        p_topic = self._softmax_vector(p_topic)
        return p_topic
# ___________________________________________________________________________

    def _store_theta(self):
        thetas_prev = (self.n_dz + self.alpha)
        thetas_p_norm = thetas_prev \
            / np.sum(thetas_prev, axis=1)[:, np.newaxis]

        thetas_current = np.zeros(shape=thetas_prev.shape)
        for doc_idx, theta_p_n in enumerate(thetas_p_norm):
            theta_current = np.random.dirichlet(theta_p_n)
            thetas_current[doc_idx, :] = theta_current
        self.hist_theta.append(np.array(thetas_current))
# ___________________________________________________________________________


def main():
    corpus = pd.read_pickle('corpus_test.pkl')
    n_it = 250
    sigma_0_sq = 1
    sigma_sq = 0.01
    delta_sq = 1
    K = 2
    autoreg = False
    dtm_alpha = DTM_Alpha(K=K, sigma_0_sq=sigma_0_sq, sigma_sq=sigma_sq,
                          delta_sq=delta_sq, autoreg=autoreg)
    beta = np.array([[0, 0, 0, 0.3, 0.7, 0, 0, 0, 0, 0],
                    [0, 0, 0.2, 0, 0, 0, 0, 0.3, 0.3, 0.2]])
    beta = []
    for idx in xrange(0, K):
        beta.append(np.random.dirichlet(np.full(10, 0.1)))
    beta = np.array(beta)

    dtm_alpha.beta = beta
    dtm_alpha.fit(n_it=n_it, corpus=corpus)
    # print dtm_alpha.hist_alpha[:-1]
    out = np.array(dtm_alpha.hist_alpha[-1:])[0]
    for idx, entry in enumerate(out):
        out[idx] = dtm_alpha._softmax_vector(entry)

    print out.T

    import matplotlib.pyplot as plt
    labels = ['0', '1']
    T = 20
    plt.figure()
    linspace_t = np.linspace(0, T-1, num=T)
    for y_arr, label in zip(out.T, labels):
        plt.plot(linspace_t, y_arr, label=label)
    plt.show()

    '''
    Save model
    '''
    # TODO: Dedicate a function
    # vars_dtm = vars(dtm_alpha)
    # with open('model.pkl', 'wb') as f:
    #   pkl.dump(vars_dtm, f)

    '''
    Load and fit
    '''
    # dtm_alpha.load_fit('model.pkl', n_it=n_it)
    # vars_dtm = vars(dtm_alpha)
    # with open('model.pkl', 'wb') as f:
    #   pkl.dump(vars_dtm, f)


if __name__ == "__main__":
    main()
