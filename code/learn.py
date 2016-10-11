
import sys
import os
import pandas as pd

HOME_PATH = os.path.expanduser('~')

sys.path.append(HOME_PATH + '/Projects/lda/code')

from lda import VariationalLDA

def main():
    corpus = pd.read_pickle('../output/corpus.pickle')
    corpus = corpus.to_dict()

    v_lda = VariationalLDA(corpus=corpus, K=10)

if __name__ == '__main__':
    main()
