import sys
import os
import pandas as pd

HOME_PATH = os.path.expanduser('~')

sys.path.append(HOME_PATH + '/Projects/lda/code')
from lda import VariationalLDA


def main():
    corpus = pd.read_pickle('../pickles/corpus.pickle')
    corpus = corpus.to_dict()

    v_lda = VariationalLDA(corpus=corpus, K=10)
    v_lda.run_vb(n_its=10)

    for i in range(0, 10):
        topic_as_dict = v_lda.get_topic_as_dict(i)

        topic_filename = 'topic_' + str(i) + '.pickle'
        topic_series = pd.Series(topic_as_dict)
        topic_series.to_pickle('../pickles/' + topic_filename)

if __name__ == '__main__':
    main()
