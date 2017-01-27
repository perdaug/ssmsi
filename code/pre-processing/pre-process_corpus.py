import os
import pandas as pd
import numpy as np

# Smaller corpus for Gibbs sampling
def preprocess_corpus(corpus, vocab, factor):
	pp_corpus = np.zeros((len(corpus), len(vocab)), dtype=np.int64)
	for doc in corpus:
		# Finding maximum intensity.
		max_intensity = 0
		for word in corpus[doc]:
			if corpus[doc][word] > max_intensity:
				max_intensity = corpus[doc][word]

		# Populating the preprocessed corpus.
		for w, word in enumerate(vocab):
			if word in corpus[doc]:
				d = int(doc) - 1
				normalised_intensity = int(corpus[doc][word] / \
					(max_intensity / factor))
				pp_corpus[d][w] = normalised_intensity
	return pp_corpus


from optparse import OptionParser
op = OptionParser()
op.add_option("--source",
							action="store", type=str,
							help="The choice of data origins.")
op.add_option("--corpus",
							action="store", type=str,
							help="The choice of a data set.")
(opts, args) = op.parse_args()


HOME_PATH = os.path.expanduser('~') + '/Projects/ssmsi/'
DATA_PATH = HOME_PATH + 'pickles/corpora/' + opts.source + '/'
AUX_DATA_PATH = HOME_PATH + 'pickles/corpora/a-to-h/'
INPUT_FILE_NAME = opts.source + '_corpus_' + opts.corpus + '.pickle'
OUT_PATH = DATA_PATH
OUTPUT_FILE_NAME = opts.source + '_corpus_' + opts.corpus + '_pp.pickle'


import pickle	
def main():
	# print(dir(pd))

	# with open(DATA_PATH + INPUT_FILE_NAME, 'rb') as f:
	# 	corpus_series = pickle.load(f)
	# print(dir(corpus_series))
	# exit()
	corpus_series = pd.read_pickle(DATA_PATH + INPUT_FILE_NAME)

	corpus = corpus_series.to_dict()
	vocab_pickle = pd.read_pickle(AUX_DATA_PATH + 'a-to-h_vocab.pickle')
	vocab = vocab_pickle.tolist()

	pp_corpus = preprocess_corpus(corpus, vocab, factor=20)
	pp_corpus.dump(OUT_PATH + OUTPUT_FILE_NAME)

if __name__ == '__main__':
	main()
