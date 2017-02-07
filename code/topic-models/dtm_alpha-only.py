import os
import pandas as pd
import numpy as np
import copy

np.set_printoptions(threshold=np.nan)

'''
References:
- Blei and Lafferty (2006)
'''

class DTM_alpha(object):

	def __init__(self, K, iter_count):
		self.iter_count = iter_count
		self.K = K
		self.history_alphas = []

	def _initialise(self, corpus):
		self.T, self.V = corpus.shape
		self.beta = np.full((self.K, self.V), 1./self.V)
		# INIT ALPHAS
		self.alphas = np.zeros(shape=(self.T, self.K))
		self.mean_alpha_init = 0
		self.var_alpha_init = 0.2
		self.var_alpha_prop = 0.05
		self.var_alpha = 0.01

		self.alphas[0] = np.random.normal(self.mean_alpha_init, 
																			self.var_alpha_init, 
																			self.K)
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
			print 'Iteration: %s.' % it
			# 2) Alpha Update
			self.history_alphas.append(copy.deepcopy(self.alphas))
			self._update_alphas()
			print self.changed
			# 3) Sampling
			self._sample_topics(corpus)

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
		self.changed = 0
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
				self.changed += 1

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

from optparse import OptionParser
op = OptionParser()
op.add_option("--source",
							action="store", type=str,
							help="The choice of data origins.")
op.add_option("--corpus",
							action="store", type=str,
							help="The choice of a data set.")
op.add_option("--iter",
							action="store", type=int,
							help="The iteration count.")
op.add_option("-K",
							action="store", type=int,
							help="The iteration count.")
(opts, args) = op.parse_args()

HOME_PATH = os.path.expanduser('~') + '/Projects/ssmsi/'
DATA_PATH = HOME_PATH + 'pickles/corpora/' + opts.source + '/'
INPUT_FILE_NAME = opts.corpus + '.pkl'
OUT_PATH = HOME_PATH + 'pickles/topics/dtm_alpha/'
OUTPUT_FILE_NAME = 'g_thetas_' + opts.source + '_' + opts.corpus \
									 + '.pickle'

def main():
	dtm_alpha = DTM_alpha(K=opts.K, iter_count=opts.iter)
	corpus = pd.read_pickle(DATA_PATH + INPUT_FILE_NAME)
	dtm_alpha.fit(corpus)
	history_alphas = dtm_alpha.history_alphas
	pickle = np.array(history_alphas)
	pickle.dump(OUT_PATH + 'alpha-history_' + str(opts.iter) 
			+ '_' + opts.corpus + '.pkl')

if __name__ == '__main__':
	main()
