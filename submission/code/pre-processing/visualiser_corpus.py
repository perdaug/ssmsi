
"""
VERSION
- Python 2

PURPOSE
- A plotting helper for the notebooks
"""

import numpy as np
import math
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import os


class Visualiser_Corpus(object):

    def __init__(self, corpus, vocab, n_rows, l_row, l_column, path_plots=None,
                 save=True, no_corpus=None, no_experiment=None):
        self.corpus = corpus
        self.no_corpus = no_corpus
        self.no_experiment = no_experiment
        self.path_plots = path_plots
        self.vocab = vocab
        self.T = len(self.corpus)
        self.V = len(self.vocab)
        self.n_rows = n_rows
        self.l_row = l_row
        self.l_column = l_column
        self.n_columns, self.map_coord = self._generate_coordinates()
        self.linspace_t = np.linspace(0, self.T - 1, num=self.T)
        self.linspace_V = np.linspace(0, self.V - 1, num=self.V)
        self.save = save

        if self.path_plots is not None:
            if not os.path.exists(self.path_plots):
                os.makedirs(self.path_plots)

# ___________________________________________________________________________
# The class's initialisation

    def _generate_coordinates(self):
        s_column, s_row = self._calc_sizes_map()
        x_coord = 0
        y_coord = self.n_rows - 1
        scanning_right = True
        map_coord = {}
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
# The class's helper functions

    def softmax(self, arr):
        return np.exp(arr) / np.sum(np.exp(arr))

    def softmax_matrix(self, arr):
        arr_softmax = np.array(arr)
        for idx_r, row in enumerate(arr):
            arr_softmax[idx_r] = np.exp(row) / np.sum(np.exp(row))
        return arr_softmax

# ___________________________________________________________________________

    '''
    Plot a vocabulary term of a pre-processed corpus
    '''
    def locate_word(self, word):
        # Mapping to the z values.
        z_matrix = np.zeros(shape=(self.n_rows, self.n_columns))
        word_idx = np.where(self.vocab == word)
        # word_idx = self.vocab.index(word)
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
        layout = go.Layout(title="The term's occurrences",
                           height=350,
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

    '''
    Plot a topic based on the inferred results
    '''
    def locate_topic(self, id_topic, thetas):
        theta = thetas[-1]
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

# __________________________________________________________________________

    '''
    Plot the theta and phi performance comparison
    '''
    def compare_performances(self, perf_ar, perf_nonar, s_batch):
        theta_ar, phi_ar = perf_ar
        theta_nonar, phi_nonar = perf_nonar
        '''
        Axis formatting
        '''
        from matplotlib.ticker import MaxNLocator
        '''
        Plotting thetas
        '''
        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.title(r'$\theta$')
        plt.plot(theta_ar, label='AR')
        plt.plot(theta_nonar, label='Non-AR')
        plt.legend(loc=2)
        plt.ylabel('Total error')
        plt.xlabel('Batch no. (s_batch = {})'.format(s_batch))
        if self.save:
            name_file = 'performance-theta_dataset-{}.png'.format(self.no_corpus)
            plt.savefig(self.path_plots + name_file, dpi=300)
        '''
        Plotting phis
        '''
        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.title(r'$\phi$')
        plt.plot(phi_ar, label='AR')
        plt.plot(phi_nonar, label='Non-AR')
        plt.legend(loc=2)
        plt.ylabel('Total error')
        plt.xlabel('Batch no. (s_batch = {})'.format(s_batch))
        if self.save:
            name_file = 'performance-phi_dataset-{}.png'.format(self.no_corpus)
            plt.savefig(self.path_plots + name_file, dpi=300)

# __________________________________________________________________________
# The plots corresponding to the inferred results

    def plot_latent_thetas(self, thetas, title):
        '''
        Axis formatting
        '''
        from matplotlib.ticker import MaxNLocator
        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        '''
        Pre-processing
        '''
        thetas = thetas.T
        '''
        Plotting
        '''
        for theta in thetas:
            plt.plot(self.linspace_t, theta)
        plt.xlabel('t')
        plt.ylabel(r'$\theta$')
        plt.title(title)
        if self.save:
            name_file = 'theta-latent_{}_dataset-{}.png'.format(title, self.no_corpus)
            plt.savefig(self.path_plots + name_file, dpi=300)

    def plot_latent_phis(self, phis, title):
        '''
        Axis formatting
        '''
        index = np.arange(10)
        bar_width = 0.2
        '''
        Plotting
        '''
        plt.figure()
        plt.title(title)
        for idx, phi in enumerate(phis):
            plt.bar(idx * bar_width + self.linspace_V, phi, bar_width)
        plt.xticks(index + bar_width, np.arange(0, 10, 1))
        plt.xlabel('v')
        plt.ylabel(r'$\phi$')
        plt.title(title)
        if self.save:
            name_file = 'phi-latent_{}_dataset-{}.png'.format(title, self.no_corpus)
            plt.savefig(self.path_plots + name_file, dpi=300)

    def plot_latent_alpha(self, alpha, title):
        '''
        Axis formatting
        '''
        from matplotlib.ticker import MaxNLocator
        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        '''
        Pre-processing
        '''
        alphas = alpha.T
        '''
        Plotting
        '''
        for alpha in alphas:
            plt.plot(self.linspace_t, alpha)
        plt.xlabel('t')
        plt.ylabel(r'$\alpha$')
        plt.title(title)
        if self.save:
            name_file = 'latent_alphas_{}.png'.format(title)
            plt.savefig(self.path_plots + name_file, dpi=300)


# ___________________________________________________________________________
# The plots corresponding to the initial corpus generation settings

    def plot_init_phi(self, betas, title):
        '''
        Axis formatting
        '''
        index = np.arange(10)
        bar_width = 0.25
        '''
        Plotting
        '''
        plt.figure()
        for idx, beta in enumerate(betas):
            plt.bar(idx * bar_width + self.linspace_V, beta, bar_width)
        plt.xticks(index, np.arange(0, 10, 1))
        plt.xlabel('v')
        plt.ylabel(r'$\phi$')
        plt.title(title)
        if self.save:
            name_file = 'phi-init.png'
            plt.savefig(self.path_plots + name_file, dpi=300)

    def plot_init_thetas(self, title, alphas=None, thetas=None):
        '''
        Integer axis labelling
        '''
        from matplotlib.ticker import MaxNLocator
        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        '''
        Pre-processing
        '''
        if alphas is not None:
            alphas_softmax = self.softmax_matrix(alphas)
            thetas = alphas_softmax
        if thetas is not None:
            thetas = thetas.T
        '''
        Plotting
        '''
        for theta in thetas:
            plt.plot(self.linspace_t, theta)
        # print(self.linspace_t)
        plt.xlabel('t')
        plt.ylabel(r'$\theta$')
        plt.title(title)
        if self.save:
            name_file = 'theta-init.png'
            plt.savefig(self.path_plots + name_file, dpi=300)

    def plot_init_alphas(self, alphas):
        if alphas is None:
            print('No alphas used.')
            return
        '''
        Integer axis labelling
        '''
        from matplotlib.ticker import MaxNLocator
        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        '''
        Pre-processing
        '''
        alphas = alphas.T
        '''
        Plotting
        '''
        for alpha in alphas:
            plt.plot(self.linspace_t, alpha)
        plt.xlabel('t')
        plt.ylabel(r'$\alpha$')
