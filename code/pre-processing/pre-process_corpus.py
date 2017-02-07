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
op.add_option("--th",
							action="store", type=float,
							help="The max count threshold.")
(opts, args) = op.parse_args()


HOME_PATH = os.path.expanduser('~') + '/Projects/ssmsi/'
DATA_PATH = HOME_PATH + 'pickles/corpora/' + opts.source + '/'
AUX_DATA_PATH = HOME_PATH + 'pickles/corpora/a-to-h/'
OUT_PATH = DATA_PATH
INPUT_FILE_NAME = opts.name + '.pkl'
OUTPUT_FILE_NAME = 'pp_' + INPUT_FILE_NAME

THRESHOLD_ND = opts.th

# Smaller corpus for Gibbs sampling
def preprocess_corpus(corpus, vocab):
	pp_corpus = np.zeros((len(corpus), len(vocab)), dtype=np.float)
	for doc in corpus:
		# Finding the maximum intensity.
		max_ND = 0
		for word in corpus[doc]:
			if corpus[doc][word] > max_ND:
				max_ND = corpus[doc][word]
		factor_norm = THRESHOLD_ND / max_ND

		# Populating the preprocessed corpus.
		for w, word in enumerate(vocab):
			if word in corpus[doc]:
				count_norm = factor_norm * corpus[doc][word]
				pp_corpus[doc][w] = int(count_norm)
	return pp_corpus

def main():
	corpus = pd.read_pickle(DATA_PATH + INPUT_FILE_NAME)
	vocab = pd.read_pickle(DATA_PATH + 'vocab_' + INPUT_FILE_NAME)
	pp_corpus = preprocess_corpus(corpus, vocab)
	pp_corpus.dump(OUT_PATH + OUTPUT_FILE_NAME)
	print('Number of words per document: ')
	print(np.sum(pp_corpus, axis=1))

if __name__ == '__main__':
	main()
