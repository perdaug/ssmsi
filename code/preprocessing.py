import os
import pandas as pd
import numpy as np

HOME_PATH = os.path.expanduser('~')

def create_vocab(corpus):
    vocab = []
    for doc in corpus:
        for word in corpus[doc]:
            if word not in vocab:
                vocab.append(word)
    vocab_panda = pd.Series(vocab)
    vocab_panda.to_pickle(HOME_PATH + '/Projects/tminm/heavy_pickles/' + 'vocab.pickle')
    print(vocab)
    exit()
    return vocab

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

    pp_corpus.dump(HOME_PATH + '/Projects/tminm/heavy_pickles/' + 'preprocessed_corpus.pickle')

def main():
    corpus_series = pd.read_pickle(HOME_PATH + '/Projects/tminm/heavy_pickles/corpus.pickle')
    corpus = corpus_series.to_dict()

    vocab = create_vocab(corpus)
    preprocess_corpus(corpus, vocab, factor=20)

if __name__ == '__main__':
    main()
    