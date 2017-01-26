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
		self.deviation_a_basic = 0.1
		self.deviation_a_guess = 3
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
			a_star_init = np.random.normal(a_init, self.deviation_a_guess,
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
					p_topic = mean_parameter(p_topic)
					z_new = np.random.choice(self.K, p=p_topic)

					self._z_history[d][w][it] = z_new
					self._dz_counts[d, z_new] += 1
					self._zw_counts[z_new, w] += 1
					self._z_counts[z_new] += 1

	def _eval_normal_multivariate(self, x, mean, cov):
		term_first = 1./((2*np.pi)**(self.K/2)*np.linalg.det(cov)**(.1/2))
		term_second = np.exp(-1./2*np.dot( \
								  np.dot(np.transpose(x-mean),np.linalg.inv(cov)),(x-mean)))
		print term_first
		return term_first*term_second

	def _dynamic_step(self):
		a_new = np.random.normal(self.alpha, self.deviation_a_basic, self.K)
		a_star_new = np.random.normal(a_new, self.deviation_a_guess, self.K)
		
		diag_dev_basic = np.full((1, self.K), self.deviation_a_basic)
		dev_basic_matrix = np.diagflat(diag_dev_basic)

		p = np.random.multivariate_normal(a_new, dev_basic_matrix)
		c = self._eval_normal_multivariate(a_new, self.alpha, dev_basic_matrix)
		print c
		print self._eval_normal_multivariate(np.array([0, 0]), np.array([0, 0]), np.array([[0.00000000001,0],[0,0.00000000001]]))
		exit()

		theta = mean_parameter(self.alpha)
		theta_star = mean_parameter(self.alpha_stars[-1])
		p = self.alpha * a_new * theta
		p_star = self.alpha_stars[-1] * a_star_new * theta_star

		p_star_u = mean_parameter(p_star)
		p_u = mean_parameter(p)
		r = p_star_u / p_u
		p_acceptance = np.minimum(r, 1)
		draws = np.zeros(shape=10)
		for idx, prob in enumerate(p_acceptance):
			choice = np.random.choice(2, p=[1 - prob, prob])
			draws[idx] = int(choice)
			if choice == 1:
				a_new[idx] = a_star_new[idx]

		self.alphas.append(a_new)
		self.alpha = a_new
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

def mean_parameter(arr):
    return np.exp(arr) / np.sum(np.exp(arr))


def main():
	dtm_alpha = DTM_alpha(K=10, iter_count=opts.iter, t_slices=opts.slices)
	corpus = pd.read_pickle(DATA_PATH + INPUT_FILE_NAME)

	dtm_alpha.fit(corpus)
	dtm_alpha.dump_thetas(OUT_PATH + 'gibbs_thetas.pickle')

if __name__ == '__main__':
	main()
