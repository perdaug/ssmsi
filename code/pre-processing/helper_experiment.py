
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
        generator_batch = self._chunks(hist_theta, s_batch)
        performance_batch_theta = []
        performance_batch_phi = []
        '''
        Calculating the diff in thetas
        '''
        for hist_theta_batch in generator_batch:
            theta_latent = np.average(hist_theta_batch, axis=0)
            # TODO: Fix theta tuning with non pre-set alphas
            matchings_topic = self._tune_thetas(alpha_init, theta_latent)
            theta_init = self._softmax_matrix(alpha_init).T
            theta_latent = theta_latent.T
            diffs_theta = []
            for idx_init, row_init in enumerate(theta_init):
                idx_latent = matchings_topic[idx_init]
                diff_theta = np.abs(row_init - theta_latent[idx_latent, :]).sum()
                diffs_theta.append(diff_theta)
            sum_theta = np.sum(diffs_theta)
            performance_batch_theta.append(sum_theta)
        '''
        Calculating the diff in phis
        '''
        hist_phi = np.array(clf.hist_phi)
        generator_batch = self._chunks(hist_phi, s_batch)
        for hist_phi_batch in generator_batch:

            hist_phi = np.array(clf.hist_phi)
            phi_latent = np.average(hist_phi_batch, axis=0)
            diffs_phi = []
            for idx_init, row_init in enumerate(beta_init):
                idx_latent = matchings_topic[idx_init]
                diff_phi = np.abs(row_init - phi_latent[idx_latent, :]).sum()
                diffs_phi.append(diff_phi)
            sum_phi = np.sum(diffs_phi)
            performance_batch_phi.append(sum_phi)
        return (performance_batch_theta, performance_batch_phi)

    def _chunks(self, l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def _tune_thetas(self, alpha_init, theta_latent):
        theta_init = self._softmax_matrix(alpha_init)
        theta_init = theta_init.T
        theta_latent = theta_latent.T
        matchings_topic = []
        for idx_init, row_init in enumerate(theta_init):
            diff_min = 1000000
            idx_match = -1
            for idx_latent, row_latent in enumerate(theta_latent):
                diff = np.abs(row_latent - row_init).sum()
                if diff < diff_min:
                    diff_min = diff
                    idx_match = idx_latent
            # print("init_{} matched to latent_{}, diff={:.2f}".format(
            #       idx_init, idx_match, diff_min))
            matchings_topic.append(idx_match)
        return matchings_topic

    def _softmax_matrix(self, arr):
        arr_softmax = np.array(arr)
        for idx_r, row in enumerate(arr):
            arr_softmax[idx_r] = np.exp(row) / np.sum(np.exp(row))
        return arr_softmax


