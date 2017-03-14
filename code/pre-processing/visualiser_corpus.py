
import numpy as np
import math
import plotly.graph_objs as go
import matplotlib.pyplot as plt
# ___________________________________________________________________________


class Visualiser_Corpus(object):

    def __init__(self, corpus, vocab, n_rows, l_row, l_column):
        self.corpus = corpus
        # self.vocab = self._extract_vocab()
        self.vocab = vocab
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
        # print(s_row)
        for j in range(0, self.n_rows):
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
        s_scan = (self.T + 1) / self.n_rows
        s_column_r = s_scan * (self.l_column / (self.l_row + self.l_column))
        s_column = int(math.floor(s_column_r))
        s_row = s_scan - s_column
        return s_column, int(s_row)
# ___________________________________________________________________________

    '''
    Used on a pre-processed corpus
    '''
    def locate_word(self, word):
        # Mapping to the z values.
        z_matrix = np.zeros(shape=(self.n_rows, self.n_columns))
        word_idx = self.vocab.index(word)
        for i, doc in enumerate(self.corpus):
            if int(i) in self.map_coord:
                doc_coord = self.map_coord[int(i)]
                z_matrix[doc_coord['y']][doc_coord['x']] = doc[word_idx]
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

# ___________________________________________________________________________

    def locate_topic(self, id_topic, thetas):
        theta = thetas[-1, :, :]
        # Mapping to the z values.
        z_matrix = np.zeros(shape=(self.n_rows, self.n_columns))
        for i, doc in enumerate(theta):
            doc_id = i
            if doc_id in self.map_coord:
                doc_coord = self.map_coord[doc_id]
                z_matrix[doc_coord['y']][doc_coord['x']] = doc[id_topic]
        # Creating the x and y axes.
        x_vector = np.zeros(shape=(self.n_rows, 1))
        for i in range(0, self.n_rows):
            x_vector[i] = i
        y_vector = np.zeros(shape=(self.n_rows, 1))
        for i in range(0, self.n_rows):
            y_vector[i] = i
        layout = go.Layout(title='\'Topic %s\' probability distribution.'
                           % id_topic,
                           height=350,
                           xaxis=dict(title='x'),
                           yaxis=dict(title='y'))
        data = [go.Heatmap(z=z_matrix,
                           x=x_vector,
                           y=y_vector,
                           colorscale='Electric',
                           colorbar=dict(title='Probability'))]
        fig = go.Figure(data=data, layout=layout)
        return fig

# ___________________________________________________________________________

    def plot_latent_alpha(self, history_alpha, n_it, T, var_init, var_basic,
                          var_prop):
        '''
        Calculate the softmax values
        '''
        labels = []
        curves = []
        settings = []
        colours = ['orange', 'b']
        for it in range(0, 2):
            idx = n_it - int(it * n_it / 2)
            alphas_last_proposed = history_alpha[idx - 1]
            alphas_softmax_last_proposed = np.zeros(shape=alphas_last_proposed.shape)
            for t, alphas_last_proposed_t in enumerate(alphas_last_proposed):
                alphas_softmax_last_proposed[t] = self.softmax(alphas_last_proposed_t)
            alphas_softmax_last_proposed = alphas_softmax_last_proposed.T
            for id_label, alpha in enumerate(alphas_softmax_last_proposed):
                curves.append(alpha)
                settings.append((colours[id_label], 1 - (it * 0.5)))
                labels.append('k=%d, it=%d' % (id_label, idx))
        '''
        Plot the figure
        '''
        from matplotlib.ticker import MaxNLocator
        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        linspace_t = np.linspace(0, T - 1, num=T)
        for y_arr, label, setting in zip(curves, labels, settings):
            plt.plot(linspace_t, y_arr, label=label, color=setting[0],
                     alpha=setting[1])
        title_fig = '$\sigma_0^2=%.2f,\quad \sigma^2=%.4f,\quad \delta^2=%.1f$' % (var_init, var_basic, var_prop)
        plt.title(title_fig)
        plt.legend(loc=1)
        plt.xlabel('t')
        plt.ylabel('$\theta$')

    def softmax(self, arr):
        return np.exp(arr) / np.sum(np.exp(arr))
