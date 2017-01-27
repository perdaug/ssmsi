import os
import pandas as pd
import numpy as np
np.set_printoptions(edgeitems=5)

'''
References:
- Blei and Lafferty (2006)
'''

class DTM_alpha(object):

	def __init__(self, K, iter_count, t_slices):
		self.iter_count = iter_count
		self.K = K
		self.t_slices = t_slices
		self.thetas = []
		self.alpha_stars = []
		self.alphas = []
		self.var_basic = 0.2
		self.var_prop = 0.5
		self.r_rates = []

	def fit(self, corpus):
		self._fit(corpus)

	def dump_thetas(self, path):
		pickle = np.array(self.thetas)
		pickle.dump(path)

	def _fit(self, corpus):
		corpus_partitions = np.array_split(corpus, self.t_slices)
		for idx_p, corpus_p in enumerate(corpus_partitions):
			self._initialise(corpus_p, idx_p)
			for it in range(self.iter_count):
				print("Time-slice: %s, Iteration: %s." % (idx_p, it))
				self._sample_topics(corpus_p)
				self._store_theta()
			self._dynamic_step()
		print(self.r_rates)
		print(sum(self.r_rates)/5)


	def _initialise(self, corpus_p, idx_p):
		self.D, self.W = corpus_p.shape
		if idx_p == 0:
			mean_a_init = 1./self.K
			deviation_a_init = 0.2
			a_init = np.random.normal(mean_a_init, deviation_a_init, self.K)
			self.alpha = a_init
			self.alphas.append(a_init)
			a_star_init = np.random.normal(a_init, self.var_prop,
																		 self.K)
			self.alpha_stars.append(a_star_init)
			self.beta = np.full((self.K, self.W), 1./self.W)

		self._zw_counts = np.zeros((self.K, self.W), dtype=np.int)
		self._dz_counts = np.zeros((self.D, self.K), dtype=np.int)
		self._z_counts = np.zeros(self.K, dtype=np.int)
		self._z_history = []

		for d, doc in enumerate(corpus_p):
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

					# Eq. 5, the second denominator is  a constant
					p_topic = ((1.0 * self._zw_counts[:, w] + self.beta[:, w]) / \
						(1.0 * self._z_counts + self.W * self.beta[:, w])) * \
						((self._dz_counts[d, :] + self.alpha))
					p_topic = _map_mp(p_topic)
					z_new = np.random.choice(self.K, p=p_topic)

					self._z_history[d][w][it] = z_new
					self._dz_counts[d, z_new] += 1
					self._zw_counts[z_new, w] += 1
					self._z_counts[z_new] += 1

	def _eval_n_dis(self, x, mean, var):
		var_diag = np.full((1, self.K), var)
		var_matrix = np.diagflat(var_diag)
		term_first = 1./((2*np.pi)**(self.K/2) \
							   *np.linalg.det(var_matrix)**(.1/2))
		term_second = np.exp(-1./2*np.dot( \
								  np.dot(np.transpose(x-mean), np.linalg.inv(var_matrix)), (x-mean)))
		return term_first*term_second

	def _eval_m_dis(self, x, p):
		# NOTE: The constant is removed (special case)
		# print p**1000
		# print x
		# exit()
		# print p, x
		p_sep = p**x
		p_prod = np.prod(p_sep)
		return p_prod

	'''
	- a stands for alpha
	- t stands for theta
	- p stands for probability
	'''
	def _dynamic_step(self):
		a_t = np.random.normal(self.alpha, self.var_basic, self.K)
		a_tplus = np.random.normal(a_t, self.var_basic, self.K)
		p_a_t = self._eval_n_dis(a_t, self.alpha, self.var_basic)
		p_a_tplus = self._eval_n_dis(a_tplus, a_t, self.var_basic)
		
		a_p_t = np.random.normal(self.alpha, self.var_prop, self.K)
		a_p_tplus = np.random.normal(a_p_t, self.var_prop, self.K)
		p_a_p_t = self._eval_n_dis(a_p_t, self.alpha, self.var_basic)
		p_a_p_tplus = self._eval_n_dis(a_p_tplus, a_p_t, self.var_basic)

		t_t = _map_mp(a_t)
		p_t_t = self._eval_m_dis(x=self._z_counts, p=t_t)

		t_p_t = _map_mp(a_p_t)
		p_t_p_t = self._eval_m_dis(x=self._z_counts, p=t_p_t)

		pa = p_a_t * p_a_tplus * p_t_t
		pa_t = p_a_p_t * p_a_p_tplus * p_t_p_t

		num_stabl = np.exp(np.log(pa) - np.log(pa_t))
		r = np.minimum(1, num_stabl)
		print(r)



		a_star_new = np.random.normal(a_t, self.var_prop, self.K)
		

		return
		# p = np.random.multivariate_normal(a_t, dev_basic_matrix)
		# c = self._eval_normal_multivariate(a_t, self.alpha, dev_basic_matrix)
		exit()

		theta = _map_mp(self.alpha)
		theta_star = _map_mp(self.alpha_stars[-1])
		p = self.alpha * a_t * theta
		p_star = self.alpha_stars[-1] * a_star_new * theta_star

		p_star_u = _map_mp(p_star)
		p_u = _map_mp(p)
		r = p_star_u / p_u
		p_acceptance = np.minimum(r, 1)
		draws = np.zeros(shape=10)
		for idx, prob in enumerate(p_acceptance):
			choice = np.random.choice(2, p=[1 - prob, prob])
			draws[idx] = int(choice)
			if choice == 1:
				a_t[idx] = a_star_new[idx]

		self.alphas.append(a_t)
		self.alpha = a_t
		self.alpha_stars.append(a_star_new)
		array = np.array(draws, dtype=int)
		print(array)
		r_rate = array.sum()/10.
		self.r_rates.append(r_rate)

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
op.add_option("--slices",
							action="store", type=int,
							help="The number of the time-slices.")
(opts, args) = op.parse_args()

HOME_PATH = os.path.expanduser('~') + '/Projects/ssmsi/'
DATA_PATH = HOME_PATH + 'pickles/corpora/' + opts.source + '/'
INPUT_FILE_NAME = opts.source + '_corpus_' + opts.corpus + '_pp.pickle'
OUT_PATH = HOME_PATH + 'pickles/topics/dtm_alpha/'
OUTPUT_FILE_NAME = 'g_thetas_' + opts.source + '_' + opts.corpus \
									 + '.pickle'

def _map_mp(arr):
    return np.exp(arr) / np.sum(np.exp(arr))


def main():
	dtm_alpha = DTM_alpha(K=10, iter_count=opts.iter, t_slices=opts.slices)
	corpus = pd.read_pickle(DATA_PATH + INPUT_FILE_NAME)

	dtm_alpha.fit(corpus)
	dtm_alpha.dump_thetas(OUT_PATH + 'gibbs_thetas.pickle')

if __name__ == '__main__':
	main()
