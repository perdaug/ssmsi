import os
import pandas as pd

HOME_PATH = os.path.expanduser('~')
PICKLES_PATH = HOME_PATH + '/Projects/ssmsi/heavy_pickles/'


def create_vocab(corpus):
	vocab = []
	for doc in corpus:
		for word in corpus[doc]:
			if word not in vocab:
				vocab.append(word)
	vocab_panda = pd.Series(vocab)
	return vocab_panda


def main():
	corpus_series = pd.read_pickle(HOME_PATH + '/Projects/tminm/heavy_pickles/corpus.pickle')
	corpus = corpus_series.to_dict()

	vocab_panda = create_vocab(corpus)
	vocab_panda.to_pickle(HOME_PATH + PICKLES_PATH + 'vocab.pickle')

if __name__ == '__main__':
	main()
	