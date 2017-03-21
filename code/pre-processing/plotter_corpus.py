
import numpy as np
import math
import plotly.graph_objs as go


class Visualiser_Corpus(object):

    def __init__(self, corpus, n_rows, l_row, l_column):
        self.corpus = corpus
        self.vocab = self._extract_vocab()
        self.T = len(self.corpus)
        self.V = len(self.vocab)
        self.n_rows = n_rows
        self.l_row = l_row
        self.l_column = l_column
        self.n_columns, self.map_coord = self._generate_coordinates()
# ___________________________________________________________________________

    def _extract_vocab(self):
        vocab = []
        for doc in self.corpus:
            for word in doc:
                if word not in vocab:
                    vocab.append(word)
        return vocab
# ___________________________________________________________________________

    def _generate_coordinates(self):
        s_column, s_row = self._calc_sizes_map()
        x_coord = 0
        y_coord = self.n_rows - 1
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

    def _calc_sizes_map(self):
        s_scan = (self.n_docs + 1) / self.n_rows
        s_column_r = s_scan * (self.l_column / (self.l_row + self.l_column))
        s_column = int(math.floor(s_column_r))
        s_row = s_scan - s_column
        return s_column, s_row
# ___________________________________________________________________________

    def locate_word(self):
        # Mapping to the z values.
        z_matrix = np.zeros(shape=(self.n_rows, self.n_columns))
        for i, doc in corpus.iteritems():
            if word_key in doc and int(i) in self.map_coord:
                doc_coord = map_coord[int(i)]
                z_matrix[doc_coord['y']][doc_coord['x']] = doc[word_key]
        # Creating the x and y axes.
        x_vector = np.zeros(shape=(self.n_columns, 1))
        for i in range(0, self.n_columns):
            x_vector[i] = i
        y_vector = np.zeros(shape=(self.n_rows, 1))
        for i in range(0, self.n_rows):
            y_vector[i] = i
        layout = go.Layout(height=350,
                           xaxis=dict(title='x'),
                           yaxis=dict(title='y'))
        data = [go.Heatmap(z=z_matrix,
                           x=x_vector,
                           y=y_vector,
                           colorscale='Electric',
                           colorbar=dict(title='Total Intensity'))]
        fig = go.Figure(data=data, layout=layout)
        return fig
