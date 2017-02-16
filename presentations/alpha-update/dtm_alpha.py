import os
import pandas as pd
import numpy as np
import copy
np.set_printoptions(threshold=np.nan)

class DTM_alpha(object):

	def __init__(self, K, iter_count, var_init, var_basic, var_prop):
		self.iter_count = iter_count
		self.K = K
		self.var_alpha_init = var_init
		self.var_alpha = var_basic
		self.var_alpha_prop = var_prop
		self.assignments_alpha = 0
		self.alpha_updates_potential = 0
		self.beta = None
		self.thetas = []


	def _initialise(self, corpus):
		self.T, self.V = corpus.shape
		if self.beta != None:
			self.beta = np.full((self.K, self.V), 1./self.V)
		# INIT ALPHAS
		self.history_alpha = np.zeros(
				shape=(self.iter_count, self.T, self.K))
		self.alphas = np.zeros(shape=(self.T, self.K))
		self.alphas[0] = np.random.normal(0, self.var_alpha_init, self.K)
		for t in xrange(1, self.T):
			self.alphas[t] = np.random.normal(self.alphas[t-1], 
					self.var_alpha)
		# INIT TOPIC ASSIGNMENTS
		self._zw_counts = np.zeros((self.K, self.V), dtype=np.int)
		self._dz_counts = np.zeros((self.T, self.K), dtype=np.int)
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

	def fit(self, corpus):
		# 1) INITIALISATION
		self._initialise(corpus)
		for it in range(self.iter_count):
			if it != 0 and it % 10 == 0:
				print 'Iteration: %s' % it
				# print 'Total alpha changes: %s' % self.assignments_alpha
				# print 'Alpha update rate: %.4f' % (1.0 \
				# 		* self.assignments_alpha / self.alpha_updates_potential)
			# 2) Alpha Update
			self.history_alpha[it] = copy.deepcopy(self.alphas)
			# self._update_alphas()
			# 3) Sampling
			self._sample_topics(corpus)
			self._store_theta()

	def eval_n_dis_log(self, x, mean, var):
		product = np.dot(np.transpose(x-mean), (x-mean))
		return -1/2 * product / var

	def _map_mp(self, arr):
		return np.exp(arr) / np.sum(np.exp(arr))

	def map_mp(self, arr, i):
		return np.exp(arr[i]) / np.sum(np.exp(arr))

	def _update_alphas(self):
		# INITIALISE PROPOSED ALPHAS
		alphas_prop = np.zeros(shape=self.alphas.shape)
		for t in xrange(0, self.T):
			alphas_prop[t] = np.random.normal(self.alphas[t-1], 
					self.var_alpha_prop)
		# UPDATE THE ALPHAS
		for t in xrange(0, self.T):
			if t != 0:
				p_alpha_cond = self.eval_n_dis_log(self.alphas[t],
						self.alphas[t-1], self.var_alpha)
				p_alpha_cond_prop = self.eval_n_dis_log(alphas_prop[t],
				 	  alphas_prop[t-1], self.var_alpha)
			else:
				p_alpha_cond = self.eval_n_dis_log(self.alphas[t],
						self.alphas[t], self.var_alpha)
				p_alpha_cond_prop = self.eval_n_dis_log(alphas_prop[t],
						alphas_prop[t], self.var_alpha)
			if t != self.T - 1:
				p_alpha_cond_nxt = self.eval_n_dis_log(self.alphas[t+1],
						self.alphas[t], self.var_alpha)
				p_alpha_cond_prop_nxt = self.eval_n_dis_log(alphas_prop[t+1],
						alphas_prop[t], self.var_alpha)
			else:
				p_alpha_cond_nxt = self.eval_n_dis_log(self.alphas[t],
						self.alphas[t], self.var_alpha)
				p_alpha_cond_prop_nxt = self.eval_n_dis_log(alphas_prop[t],
						alphas_prop[t], self.var_alpha)

			theta_t = np.zeros(shape=self.alphas[t].shape)
			theta_prop_t = np.zeros(shape=alphas_prop[t].shape)
			for k in range(0, self.K):
				theta_t[k] = self.map_mp(self.alphas[t], k)
				theta_prop_t[k] = self.map_mp(alphas_prop[t], k)
			p_theta_t = np.sum(self._dz_counts[t] * np.log(theta_t))
			p_theta_prop_t = np.sum(self._dz_counts[t] * np.log(
					theta_prop_t))
			p_alpha_t = p_alpha_cond + p_alpha_cond_nxt + p_theta_t
			p_alpha_prop_t = p_alpha_cond_prop + p_alpha_cond_prop_nxt \
					+ p_theta_prop_t
			r_raw = np.exp(p_alpha_prop_t - p_alpha_t)
			r = np.minimum(1, r_raw)
			accept_alpha = np.random.binomial(1, r)
			if accept_alpha:
				self.alphas[t] = alphas_prop[t]
				self.assignments_alpha += 1
			self.alpha_updates_potential += 1

	def _sample_topics(self, corpus):
		for d, doc in enumerate(corpus):
			for w, _ in enumerate(doc):
				for it, z in enumerate(self._z_history[d][w]):
					self._dz_counts[d, z] -= 1
					self._zw_counts[z, w] -= 1
					self._z_counts[z] -= 1

					# Eq. 5, the second denominator is  a constant
					p_topic = ((1.0 * self._zw_counts[:, w] + self.beta[:, w]) / \
						(1.0 * self._z_counts + self.V * self.beta[:, w])) * \
						((self._dz_counts[d, :] + self.alphas[d]))
					p_topic = self._map_mp(p_topic)
					z_new = np.random.choice(self.K, p=p_topic)
					self._z_history[d][w][it] = z_new
					self._dz_counts[d, z_new] += 1
					self._zw_counts[z_new, w] += 1
					self._z_counts[z_new] += 1

	def _store_theta(self):
		thetas_prev = (self._dz_counts + self.alphas)
		thetas_p_norm = thetas_prev \
										/ np.sum(thetas_prev, axis=1)[:, np.newaxis]

		thetas_current = np.zeros(shape=thetas_prev.shape)
		for doc_idx, theta_p_n in enumerate(thetas_p_norm):
			theta_current = np.random.dirichlet(theta_p_n)
			thetas_current[doc_idx, :] = theta_current
		self.thetas.append(np.array(thetas_current))


corpus = pd.read_pickle('corpus_test.pkl')
iter_count = 500
var_init = 1
var_basic = 0.01
var_prop = 0.5
K = 2
dtm_alpha = DTM_alpha(K=K, iter_count=iter_count, var_init=var_init, var_basic=var_basic, var_prop=var_prop)
beta = np.array([[0, 0, 0, 0.3, 0.7, 0, 0, 0, 0, 0], [0, 0, 0.2, 0, 0, 0, 0, 0.3, 0.3, 0.2]])
dtm_alpha.beta = beta
dtm_alpha.fit(corpus)
thetas = np.array(dtm_alpha.thetas)
thetas.dump('thetas.pkl')
