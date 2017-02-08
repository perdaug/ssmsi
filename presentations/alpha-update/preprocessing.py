import numpy as np
np.set_printoptions(threshold=np.nan)

# Smaller corpus for Gibbs sampling
def preprocess_corpus(corpus, vocab, threshold, normalise):
	pp_corpus = np.zeros((len(corpus), len(vocab)), dtype=np.float)
	factor_norm = 1
	for doc in corpus:
		# Finding the maximum intensity.
		if normalise:
			max_ND = 0
			for word in corpus[doc]:
				if corpus[doc][word] > max_ND:
					max_ND = corpus[doc][word]
			factor_norm = threshold / max_ND

		# Populating the preprocessed corpus.
		for w, word in enumerate(vocab):
			if word in corpus[doc]:
				count_norm = factor_norm * corpus[doc][word]
				pp_corpus[doc][w] = int(count_norm)
	return pp_corpus
	