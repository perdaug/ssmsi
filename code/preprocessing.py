import os
import pandas as pd

HOME_PATH = os.path.expanduser('~')

def create_vocab(corpus):
    vocab = []
    for doc in corpus:
        for word in corpus[doc]:
            if word not in vocab:
                vocab.append(word)
    vocab_panda = pd.Series(vocab)
    vocab_panda.to_pickle(HOME_PATH + '/Projects/tminm/heavy_pickles/' + 'vocab.pickle')

def create_normalised_corpus(corpus, factor):
    for doc in corpus:
        max_intensity = 0
        for word in corpus[doc]:
            if corpus[doc][word] > max_intensity:
                max_intensity = corpus[doc][word]
        for word in corpus[doc]:
            corpus[doc][word] = int(corpus[doc][word] / \
                (max_intensity / factor))

    corpus_panda = pd.Series(corpus)
    corpus_panda.to_pickle(HOME_PATH + '/Projects/tminm/heavy_pickles/' + 'normalised_corpus.pickle')

def main():
    corpus_series = pd.read_pickle(HOME_PATH + '/Projects/tminm/heavy_pickles/corpus.pickle')
    corpus = corpus_series.to_dict()

    # create_vocab(corpus)
    create_normalised_corpus(corpus=corpus, factor=100)

if __name__ == '__main__':
    main()