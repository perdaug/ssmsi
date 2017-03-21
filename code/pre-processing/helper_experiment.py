
"""
VERSION
- Python 2

FUNCTION
- Functions to be used during the experiments
"""

import numpy as np


class Helper_Experiment(object):

    def calculate_performance(self, alpha_init, beta_init, theta_init,
                              clf, s_batch):
        hist_theta = np.array(clf.hist_theta)
        theta_init = theta_init.T
        generator_batch = self._split_to_batches(hist_theta, s_batch)
        performance_batch_theta = []
        performance_batch_phi = []
        '''
        Iterate through the theta batches
        '''
        for hist_theta_batch in generator_batch:
            '''
            Pre-process and tune the thetas
            '''
            theta_latent = np.average(hist_theta_batch, axis=0)
            if alpha_init is not None:
                theta_init = self._softmax_matrix(alpha_init)
            theta_latent = theta_latent.T
            matchings_topic = self._tune_thetas(theta_init, theta_latent)
            '''
            Calculate the theta similarity
            '''
            diffs_theta = []
            for idx_init, row_init in enumerate(theta_init):
                idx_latent = matchings_topic[idx_init]
                diff_theta = np.abs(row_init - theta_latent[idx_latent, :])
                diff_theta = diff_theta.sum()
                diffs_theta.append(diff_theta)
            sum_theta = np.sum(diffs_theta)
            performance_batch_theta.append(sum_theta)
        '''
        Iterate through the phi batches
        '''
        hist_phi = np.array(clf.hist_phi)
        generator_batch = self._split_to_batches(hist_phi, s_batch)
        for hist_phi_batch in generator_batch:
            phi_latent = np.average(hist_phi_batch, axis=0)
            '''
            Calculate the phi similarity
            '''
            diffs_phi = []
            for idx_init, row_init in enumerate(beta_init):
                idx_latent = matchings_topic[idx_init]
                diff_phi = np.abs(row_init - phi_latent[idx_latent, :])
                diff_phi = diff_phi.sum()
                diffs_phi.append(diff_phi)
            sum_phi = np.sum(diffs_phi)
            performance_batch_phi.append(sum_phi)
        return (performance_batch_theta, performance_batch_phi)

    def _split_to_batches(self, arr, s_batch):
        for i in range(0, len(arr), s_batch):
            yield arr[i:i + s_batch]

    def _tune_thetas(self, theta_init, theta_latent):
        '''
        Create the matrix of all distances
        '''
        matrix_dist = []
        for idx_init, row_init in enumerate(theta_init):
            row = []
            for idx_latent, row_latent in enumerate(theta_latent):
                diff = np.abs(row_latent - row_init).sum()
                row.append(diff)
            matrix_dist.append(row)
        '''
        - Create all possible permutations
        - Assign the dist of each perm to a dict
        '''
        ids_topic = np.arange(len(theta_init))
        import itertools
        perm_all = list(itertools.permutations(ids_topic))
        perm_dict = {}
        for perm in perm_all:
            sum_perm = 0
            for idx_init, idx_latent in enumerate(perm):
                sum_perm += matrix_dist[idx_init][idx_latent]
            perm_dict[perm] = sum_perm
        '''
        Find the best permutation
        '''
        perm_best = min(perm_dict, key=perm_dict.get)
        matchings_topic = list(perm_best)
        return matchings_topic

    def _softmax_matrix(self, arr):
        arr_softmax = np.array(arr)
        for idx_r, row in enumerate(arr):
            arr_softmax[idx_r] = np.exp(row) / np.sum(np.exp(row))
        return arr_softmax
