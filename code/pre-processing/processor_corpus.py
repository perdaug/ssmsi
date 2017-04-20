
"""
VERSION
- Python 2

FUNCTION
- Corpus pre-processing
"""

import numpy as np
import math


class Processor_Corpus(object):

    def __init__(self, corpus, vocab=None):
        self.corpus = corpus
        self.vocab = vocab
        if self.vocab is None:
            self.vocab = self._extract_vocab()
        self.T = len(self.corpus)
        self.V = len(self.vocab)

    def _extract_vocab(self):
        vocab = []
        for doc in self.corpus.values():
            for word in doc.keys():
                if word not in vocab:
                    vocab.append(word)
        return np.array(vocab)
# ___________________________________________________________________________

    '''
    Terms:
    - pp: pre-processed
    '''
    def process_corpus(self, threshold, normalise):
        corpus_pp = np.zeros((self.T, self.V), dtype=np.float)
        for t, doc in self.corpus.items():
            t = int(t)
            '''
            Assign the normalisation factor based on the max intensity
            '''
            if normalise:
                max_ND = 0
                for word in doc:
                    if doc[word] > max_ND:
                        max_ND = doc[word]
                factor_norm = threshold / max_ND
            else:
                factor_norm = 1.
            '''
            Transforming the corpus into a numpy array
            '''
            for w, word in enumerate(self.vocab):
                if word in doc:
                    count_norm = factor_norm * doc[word]
                    corpus_pp[t][w] = int(count_norm)
        return corpus_pp
# ___________________________________________________________________________

    '''
    Generate the coordinates for the 2-dimensional bne plotting.
    '''
    def generate_coordinates(self, n_rows, l_row, l_column):
        s_column, s_row = self.calc_sizes(self.T, n_rows, l_row, l_column)
        x_coord = 0
        y_coord = n_rows - 1
        scanning_right = True
        map_coord = {}
        for j in range(0, n_rows):
            for i in range(0, s_row):
                doc_id = i + s_column * j + s_row * j
                map_coord[doc_id] = {'x': x_coord, 'y': y_coord}
                # Switching horizontal scanning to vertical.
                if (i % (s_row - 1) == 0 and i != 0):
                    y_coord -= 1
                    continue
                if scanning_right:
                    x_coord += 1
                else:
                    x_coord -= 1
            # Reversing the scanner's direction.
            scanning_right = not scanning_right
        return s_row, map_coord

    def calc_sizes(self, n_docs, n_rows, l_row, l_column):
        s_scan = (n_docs + 1) / n_rows
        s_column = int(math.floor(s_scan * (l_column / (l_row + l_column))))
        s_row = s_scan - s_column
        return s_column, s_row
