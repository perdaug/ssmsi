
import numpy as np
import math
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import os
# ___________________________________________________________________________


class Visualiser_Corpus(object):

    def __init__(self, corpus, vocab, n_rows, l_row, l_column,
                 path_plots=None, save=True):
        self.corpus = corpus
        # self.vocab = self._extract_vocab()
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
    '''
    TODO: Enable to vocab creation
    '''

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
    def softmax(self, arr):
        return np.exp(arr) / np.sum(np.exp(arr))

    def softmax_matrix(self, arr):
        arr_softmax = np.array(arr)
        for idx_r, row in enumerate(arr):
            arr_softmax[idx_r] = np.exp(row) / np.sum(np.exp(row))
        return arr_softmax

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


# __________________________________________________________________________

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
            plt.savefig(self.path_plots + 'performance_thetas.png', dpi=300)
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
            plt.savefig(self.path_plots + 'performance_phis.png', dpi=300)

# __________________________________________________________________________
# LATENT PLOTS

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
            name_file = 'latent_thetas_{}.png'.format(title)
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
        plt.xticks(index + bar_width, np.arange(0, 10, 1.0))
        plt.xlabel('v')
        plt.ylabel(r'$\phi$')
        plt.title(title)
        if self.save:
            name_file = 'latent_phis_{}.png'.format(title)
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
# INIT PLOTS

    def plot_init_betas(self, betas):
        '''
        Axis formatting
        '''
        index = np.arange(10)
        bar_width = 0.2
        '''
        Plotting
        '''
        plt.figure()
        for idx, beta in enumerate(betas):
            plt.bar(idx * bar_width + self.linspace_V, beta, bar_width)
        plt.xticks(index + bar_width, np.arange(0, 10, 1.0))
        plt.xlabel('v')
        plt.ylabel(r'$\beta$')

    def plot_init_thetas(self, alphas=None, thetas=None):
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
            thetas = alphas_softmax.T
        if thetas is not None:
            thetas = thetas.T
        '''
        Plotting
        '''
        for theta in thetas:
            plt.plot(self.linspace_t, theta)
        plt.xlabel('t')
        plt.ylabel(r'$\theta$')

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

# __________________________________________________________________________
# BACKLOG

    def plot_latent_alpha2(self, history_alpha, n_it, var_init, var_basic,
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
        for y_arr, label, setting in zip(curves, labels, settings):
            plt.plot(self.linspace_t, y_arr, label=label, color=setting[0],
                     alpha=setting[1])
        title_fig = '$\sigma_0^2=%.2f,\quad \sigma^2=%.4f,\quad \delta^2=%.1f$' % (var_init, var_basic, var_prop)
        plt.title(title_fig)
        plt.legend(loc=1)
        plt.xlabel('t')
        plt.ylabel(r'$\theta$')
