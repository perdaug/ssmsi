import os
import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.nan)
from optparse import OptionParser

op = OptionParser()
op.add_option("--source",
							action="store", type=str,
							help="The choice of data origins.")
op.add_option("--name",
							action="store", type=str,
							help="The choice of a data set.")
(opts, args) = op.parse_args()


HOME_PATH = os.path.expanduser('~') + '/Projects/ssmsi/'
DATA_PATH = HOME_PATH + 'pickles/corpora/' + opts.source + '/'
AUX_DATA_PATH = HOME_PATH + 'pickles/corpora/a-to-h/'
OUT_PATH = DATA_PATH
INPUT_FILE_NAME = opts.name + '.pkl'
OUTPUT_FILE_NAME = 'pp_' + INPUT_FILE_NAME

# Smaller corpus for Gibbs sampling
def preprocess_corpus(corpus, vocab, factor):
	# pp_corpus = np.zeros((len(corpus), len(vocab)), dtype=np.int64)
	pp_corpus = np.zeros((len(corpus), len(vocab)), dtype=np.float)
	for doc in corpus:
		# Finding the maximum intensity.
		max_intensity = 0
		for word in corpus[doc]:
			if corpus[doc][word] > max_intensity:
				max_intensity = corpus[doc][word]
		# Populating the preprocessed corpus.
		for w, word in enumerate(vocab):
			if word in corpus[doc]:
				normalised_intensity = float(corpus[doc][word] / \
					(max_intensity * factor))
				pp_corpus[doc][w] = normalised_intensity
	return pp_corpus

def main():
	corpus = pd.read_pickle(DATA_PATH + INPUT_FILE_NAME)
	vocab_pickle = pd.read_pickle(AUX_DATA_PATH + 'a-to-h_vocab.pkl')
	vocab = vocab_pickle.tolist()

	pp_corpus = preprocess_corpus(corpus, vocab, factor=0.1)
	pp_corpus.dump(OUT_PATH + OUTPUT_FILE_NAME)
	print('Number of words in documents: ')
	print(np.sum(pp_corpus, axis=1))
	print pp_corpus

if __name__ == '__main__':
	main()
