import os
import pandas as pd
import numpy as np
import copy
from scipy.stats import norm

np.set_printoptions(threshold=np.nan)

class dtm_autoreg(object):

	'''
	Parameters:
	- sigma_0_sq is the initial variance for alpha[0];
	- sigma_sq is the variance for alpha[t]
	- delta_sq is the variance for alpha'[t]
	'''
	def __init__(self, K, n_it, sigma_0_sq, sigma_sq, delta_sq):
		self.n_it = n_it
		self.K = K
		self.sigma_0_sq = sigma_0_sq
		self.sigma_sq = sigma_sq
		self.delta_sq = delta_sq

		self.n_upd_alpha = 0
		self.n_it_alpha = 0
		self.beta = None
		self.thetas = []

	def fit(self, corpus):
		self._initialise(corpus)
		for it in range(self.n_it):
			if it != 0 and it % 10 == 0:
				self._print_update(it)
			self.hist_alpha[it] = copy.deepcopy(self.alpha)
			self._update_alpha()
			self._sample_topics(corpus)
			self._store_theta()

	def _print_update(self, it):
		print 'Iteration: %s' % it
		print 'Total alpha changes: %s' % self.n_upd_alpha
		print 'Alpha update rate: %.4f' \
				% (1.0 * self.n_upd_alpha / self.n_it_alpha)
# __________________________________________________________________

	def _initialise(self, corpus):
		# Initialise the parameters
		self.T, self.V = corpus.shape
		self.hist_alpha = np.zeros(shape=(self.n_it,self.T,self.K))
		self.alpha = np.zeros(shape=(self.T,self.K))
		self.alpha[0] = np.random.normal(0,self.sigma_0_sq,self.K)
		for t in xrange(1, self.T):
			self.alpha[t] = np.random.normal(self.alpha[t-1], \
					self.sigma_sq)
		self.n_zw = np.zeros((self.K,self.V), dtype=np.int)
		self.n_dz = np.zeros((self.T,self.K), dtype=np.int)
		self.n_z = np.zeros(self.K, dtype=np.int)
		self.hist_z = []

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
# __________________________________________________________________

	'''
	p(z_k,a_k|x)=p(x|z_k,a_k)*p(z_k|a_k)*p(a_k):
	- First term (t1): Mult(softmax(a)_k)
	- Second term (t2): softmax(a)_k
	- Third term (t3): p(a_{k,t}|a_{k,t-1})*p(a_{k,t+1}|a_{k,t})
	-- t31: p(a_{k,t}|a_{k,t-1})
	-- t32: p(a_{k,t+1}|a_{k,t})

	Syntax:
	p - Probability
	'''	
	def _update_alpha(self):
		for t in xrange(0, self.T):
			alpha_prop = np.random.normal(self.alpha[t], self.delta_sq)
			alpha_updated = copy.deepcopy(self.alpha[t])
			for k in xrange(0, self.K):
				theta_prop = self._softmax_log(alpha_prop, k)
				t1_prop = self.n_dz[t][k] * theta_prop
				t2_prop = theta_prop
				t3_prop = self._create_t3(alpha_prop[k], t, k)
				p_alpha_prop = t1_prop + t2_prop + t3_prop

				theta = self._softmax_log(self.alpha[t], k)
				t1_prop = self.n_dz[t][k] * theta
				t2_prop = theta
				t3_prop = self._create_t3(self.alpha[t][k], t, k)
				p_alpha = t1_prop + t2_prop + t3_prop
				
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
	def _create_t3(self, alpha_prop, t, k):
		if t == 0:
			t31 = self._eval_norm(alpha_prop, 0, self.sigma_0_sq)
		else:
			t31 = self._eval_norm(alpha_prop, self.alpha[t-1][k], \
					self.sigma_sq)
		if t == self.T - 1:
			t3 = t31
		else:
			t32 = self._eval_norm(self.alpha[t+1][k], alpha_prop, \
					self.sigma_sq)
			t3 = t31 + t32
		return t3
# __________________________________________________________________

	def _sample_topics(self, corpus):
		for d, doc in enumerate(corpus):
			for w, _ in enumerate(doc):
				for it, z in enumerate(self.hist_z[d][w]):
					self.n_dz[d, z] -= 1
					self.n_zw[z, w] -= 1
					self.n_z[z] -= 1

					# Eq. 5, the second denominator is  a constant
					p_topic = ((1.0 * self.n_zw[:, w] + self.beta[:, w]) / \
						(1.0 * self.n_z + self.V * self.beta[:, w])) * \
						((self.n_dz[d, :] + self.alpha[d]))
					p_topic = self.softmax_all(p_topic)
					z_new = np.random.choice(self.K, p=p_topic)
					self.hist_z[d][w][it] = z_new
					self.n_dz[d, z_new] += 1
					self.n_zw[z_new, w] += 1
					self.n_z[z_new] += 1


	def softmax_all(self, arr):
		return np.exp(arr) / np.sum(np.exp(arr))
# __________________________________________________________________

	def _store_theta(self):
		thetas_prev = (self.n_dz + self.alpha)
		thetas_p_norm = thetas_prev \
										/ np.sum(thetas_prev, axis=1)[:, np.newaxis]

		thetas_current = np.zeros(shape=thetas_prev.shape)
		for doc_idx, theta_p_n in enumerate(thetas_p_norm):
			theta_current = np.random.dirichlet(theta_p_n)
			thetas_current[doc_idx, :] = theta_current
		self.thetas.append(np.array(thetas_current))
# __________________________________________________________________


# corpus = pd.read_pickle('corpus_test.pkl')
# n_it = 500
# sigma_0_sq = 1
# sigma_sq = 0.01
# delta_sq = 0.5
# K = 2
# dtm_alpha = DTM_alpha(K=K, n_it=n_it, sigma_0_sq=sigma_0_sq, sigma_sq=sigma_sq, delta_sq=delta_sq)
# beta = np.array([[0, 0, 0, 0.3, 0.7, 0, 0, 0, 0, 0], [0, 0, 0.2, 0, 0, 0, 0, 0.3, 0.3, 0.2]])
# dtm_alpha.beta = beta
# dtm_alpha.fit(corpus)
# thetas = np.array(dtm_alpha.thetas)
# thetas.dump('thetas.pkl')
