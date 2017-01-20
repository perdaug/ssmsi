import os
import pandas as pd
import numpy as np
np.set_printoptions(edgeitems=5)

'''
References:
- Griffiths and Stevyers (2004)
'''

class LDA_Gibbs(object):

	def __init__(self, K, iter_count):
		self.iter_count = iter_count
		self.K = K
		self.thetas = []

	def fit(self, corpus):
		self._fit(corpus)

	def dump_thetas(self, path):
		pickle = np.array(self.thetas)
		pickle.dump(path)

	def _fit(self, corpus):
		self._initialise(corpus)

		for it in range(self.iter_count):
			print("Iteration %s:" % it)
			self._sample_topics(corpus)
			self._store_theta()
			print("Current theta:")
			print(self.thetas[it])

	def _initialise(self, corpus):
		self.D, self.W = corpus.shape
		self._total_frequencies = corpus.sum()
		self.alpha = 1. / self.K
		self.beta = 1. / self.W

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

					# Eq. 5, the second denominator is  a constant
					p_topic = ((1.0 * self._zw_counts[:, w] + self.beta) / \
						(1.0 * self._z_counts + self.W * self.beta)) * \
						((self._dz_counts[d, :] + self.alpha))
					p_topic = p_topic / p_topic.sum()
					z_new = np.random.choice(self.K, p=p_topic)

					self._z_history[d][w][it] = z_new
					self._dz_counts[d, z_new] += 1
					self._zw_counts[z_new, w] += 1
					self._z_counts[z_new] += 1

	def _store_theta(self):
		thetas_prev = (self._dz_counts + self.alpha)
		thetas_p_norm = thetas_prev \
										/ np.sum(thetas_prev, axis=1)[:, np.newaxis]

		thetas_current = np.zeros(shape=thetas_prev.shape)
		for doc_idx, theta_p_n in enumerate(thetas_p_norm):
			theta_current = np.random.dirichlet(theta_p_n)
			thetas_current[doc_idx, :] = theta_current
		self.thetas.append(np.array(thetas_current))


HOME_PATH = os.path.expanduser('~') + '/Projects/ssmsi/'
DATA_PATH = HOME_PATH + 'pickles/corpora/a-to-h/'
OUT_PATH = HOME_PATH + 'pickles/results/lda_gibbs/'

def main():
	lda_gibbs = LDA_Gibbs(K=10, iter_count=50)
	# To-do: Fix the main corpus (the number of col is arbitrary) 
	corpus = pd.read_pickle(DATA_PATH + 'a-to-h_corpus_r.pickle')
	lda_gibbs.fit(corpus)
	lda_gibbs.dump_thetas(OUT_PATH + 'gibbs_thetas.pickle')

if __name__ == '__main__':
	main()
