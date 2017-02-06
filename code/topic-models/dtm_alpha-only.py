import os
import pandas as pd
import numpy as np
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
		# self.thetas = []
		# self.alpha_stars = []
		# self.alphas = []
		# self.var_basic = 0.2
		# self.r_rates = []

	def fit(self, corpus):
		self._fit(corpus)

	def dump_thetas(self, path):
		pickle = np.array(self.thetas)
		pickle.dump(path)

	def _fit(self, corpus):
		# 1) INITIALISATION
		self._initialise(corpus)
		for it in range(self.iter_count):
			print 'Iteration: %s.' % it
			# 2) Alpha Update
			import copy
			self.history_alphas.append(copy.deepcopy(self.alphas))
			self._update_alphas()
			print self.changed
			# 3) Sampling
			self._sample_topics(corpus)
			# TODO: Stores thetas
			# self._store_theta()

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
		# TODO: METROPOLIS HASTINGS
		# a_init = np.random.normal(mean_a_init, deviation_a_init, self.K)
		# self.alpha = a_init
		# self.alphas.append(a_init)
		# a_star_init = np.random.normal(a_init, self.var_alpha_prop,
		# 															 self.K)
		# self.alpha_stars.append(a_star_init)

	# def _eval_n_dis(self, x, mean, var):
	# 	var_diag = np.full((1, self.K), var)
	# 	var_matrix = np.diagflat(var_diag)
	# 	term_first = 1./((2*np.pi)**(self.K/2) \
	# 							 *np.linalg.det(var_matrix)**(.1/2))
	# 	term_second = np.exp(-1./2*np.dot( \
	# 								np.dot(np.transpose(x-mean), np.linalg.inv(var_matrix)), (x-mean)))
	# 	return term_first*term_second

	def eval_n_dis_log(self, x, mean, var):
		product = np.dot(np.transpose(x-mean), (x-mean))
		return -1/2 * product / var

	# def _eval_m_dis(self, x, p):
	# 	# NOTE: The constant is removed (special case)
	# 	p_sep = p**x
	# 	p_prod = np.prod(p_sep)
	# 	return p_sep
	# 	return p_prod

	# def eval_m_dis_log(self, x, p):
	# 	# NOTE: The constant is removed (special case)
	# 	p_sep = p**x
	# 	p_prod = np.prod(p_sep)
	# 	return p_sep
	# 	return p_prod

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
			p_theta_t = np.sum(np.log(theta_t**self._dz_counts[t]))
			p_theta_prop_t = np.sum(np.log(
					theta_prop_t**self._dz_counts[t]))
			p_alpha_t = p_alpha_cond + p_alpha_cond_nxt + p_theta_t
			p_alpha_prop_t = p_alpha_cond_prop + p_alpha_cond_prop_nxt \
					+ p_theta_prop_t
			r_raw = np.exp(p_alpha_prop_t - p_alpha_t)
			r = np.minimum(1, r_raw)
			accept_alpha = np.random.binomial(1, r)
			# self.t = 0
			if accept_alpha:
				# self.t = t
				# print self.alphas[t] == alphas_prop[t]
				self.alphas[t] = alphas_prop[t]
				# print self.alphas[t] == alphas_prop[t]

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



	'''
	- a stands for alpha
	- t stands for theta
	- p stands for probability
	'''
	# def _dynamic_step(self):
	# 	a_t = np.random.normal(self.alpha, self.var_basic, self.K)
	# 	a_tplus = np.random.normal(a_t, self.var_basic, self.K)
	# 	p_a_t = self._eval_n_dis(a_t, self.alpha, self.var_basic)
	# 	p_a_tplus = self._eval_n_dis(a_tplus, a_t, self.var_basic)
		
	# 	a_p_t = np.random.normal(self.alpha, self.var_prop, self.K)
	# 	a_p_tplus = np.random.normal(a_p_t, self.var_prop, self.K)
	# 	p_a_p_t = self._eval_n_dis(a_p_t, self.alpha, self.var_basic)
	# 	p_a_p_tplus = self._eval_n_dis(a_p_tplus, a_p_t, self.var_basic)

	# 	t_t = _map_mp(a_t)
	# 	p_t_t = self._eval_m_dis(x=self._z_counts, p=t_t)

	# 	t_p_t = _map_mp(a_p_t)
	# 	p_t_p_t = self._eval_m_dis(x=self._z_counts, p=t_p_t)

	# 	pa = p_a_t * p_a_tplus * p_t_t
	# 	pa_t = p_a_p_t * p_a_p_tplus * p_t_p_t

	# 	num_stabl = np.exp(np.log(pa) - np.log(pa_t))
	# 	r = np.minimum(1, num_stabl)
	# 	print(r)



	# 	a_star_new = np.random.normal(a_t, self.var_prop, self.K)
		

	# 	return
	# 	# p = np.random.multivariate_normal(a_t, dev_basic_matrix)
	# 	# c = self._eval_normal_multivariate(a_t, self.alpha, dev_basic_matrix)
	# 	exit()

	# 	theta = _map_mp(self.alpha)
	# 	theta_star = _map_mp(self.alpha_stars[-1])
	# 	p = self.alpha * a_t * theta
	# 	p_star = self.alpha_stars[-1] * a_star_new * theta_star

	# 	p_star_u = _map_mp(p_star)
	# 	p_u = _map_mp(p)
	# 	r = p_star_u / p_u
	# 	p_acceptance = np.minimum(r, 1)
	# 	draws = np.zeros(shape=10)
	# 	for idx, prob in enumerate(p_acceptance):
	# 		choice = np.random.choice(2, p=[1 - prob, prob])
	# 		draws[idx] = int(choice)
	# 		if choice == 1:
	# 			a_t[idx] = a_star_new[idx]

	# 	self.alphas.append(a_t)
	# 	self.alpha = a_t
	# 	self.alpha_stars.append(a_star_new)
	# 	array = np.array(draws, dtype=int)
	# 	print(array)
	# 	r_rate = array.sum()/10.
	# 	self.r_rates.append(r_rate)

	def _store_theta(self):
		thetas_prev = (self._dz_counts + self.alpha)
		thetas_p_norm = thetas_prev \
										/ np.sum(thetas_prev, axis=1)[:, np.newaxis]

		thetas_current = np.zeros(shape=thetas_prev.shape)
		for doc_idx, theta_p_n in enumerate(thetas_p_norm):
			theta_current = np.random.dirichlet(theta_p_n)
			thetas_current[doc_idx, :] = theta_current
		self.thetas.append(np.array(thetas_current))


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
(opts, args) = op.parse_args()

HOME_PATH = os.path.expanduser('~') + '/Projects/ssmsi/'
DATA_PATH = HOME_PATH + 'pickles/corpora/' + opts.source + '/'
INPUT_FILE_NAME = opts.corpus + '.pkl'
OUT_PATH = HOME_PATH + 'pickles/topics/dtm_alpha/'
OUTPUT_FILE_NAME = 'g_thetas_' + opts.source + '_' + opts.corpus \
									 + '.pickle'

def main():
	dtm_alpha = DTM_alpha(K=10, iter_count=opts.iter)
	corpus = pd.read_pickle(DATA_PATH + INPUT_FILE_NAME)
	dtm_alpha.fit(corpus)
	print len(dtm_alpha.history_alphas)
	history_alphas = dtm_alpha.history_alphas
	pickle = np.array(history_alphas)
	pickle.dump(OUT_PATH + 'history_alphas.pkl')
	# TODO: DUMP THETAS
	# dtm_alpha.dump_thetas(OUT_PATH + 'gibbs_thetas.pickle')

if __name__ == '__main__':
	main()
