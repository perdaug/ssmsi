import os
import pandas as pd
import numpy as np
from scipy.special import psi
np.set_printoptions(suppress=True)
HOME_PATH = os.path.expanduser('~')

meanchangethresh = 0.001

class Variational_LDA(object):

    def __init__(self, corpus, K, alpha, eta, tau0, kappa):
        self.corpus = corpus
        self.vocabulary = create_vocabulary(corpus)
        self._K = K
        self._W = len(self.vocabulary)
        self._alpha = alpha
        self._eta = eta
        self._tau0 = tau0 + 1
        self._kappa = kappa
        self._updatect = 0


        self._lambda = np.random.gamma(100., 1./100., (self._K, self._W))
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = np.exp(self._Elogbeta)

    def update_lambda_docs(self):
        self._rhot = pow(self._tau0 + self._updatect, -self._kappa)

        (gamma, sstats) = self.do_e_step(self.corpus, self.vocabulary)

        # INVESTIGATE D
        self._lambda = self._lambda * (1-self._rhot) + \
            self._rhot * (self._eta + sstats / len(self.corpus))
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = np.exp(self._Elogbeta)
        self._updatect += 1

        return gamma


    def do_e_step(self, corpus, vocabulary):

        (wordids, wordcts) = parse_doc_list(corpus, vocabulary)

        gamma = 1*np.random.gamma(100., 1./100., (len(corpus), self._K))
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = np.exp(Elogtheta)

        sstats = np.zeros(self._lambda.shape)

        meanchange = 0
        for d in range(0, len(corpus)):
            ids = wordids[d]
            cts = wordcts[d]
            gammad = gamma[d, :]
            Elogthetad = Elogtheta[d, :]
            expElogthetad = expElogtheta[d, :]
            expElogbetad = self._expElogbeta[:, ids]
            phinorm = np.dot(expElogthetad, expElogbetad) + 1e-100
            for it in range(0, 100):
                lastgamma = gammad
                gammad = self._alpha + expElogthetad * \
                    np.dot(cts / phinorm, expElogbetad.T)
                Elogthetad = dirichlet_expectation(gammad)
                expElogthetad = np.exp(Elogthetad)
                phinorm = np.dot(expElogthetad, expElogbetad) + 1e-100
                # If gamma hasn't changed much, we're done.
                meanchange = np.mean(abs(gammad - lastgamma))
                if (meanchange < meanchangethresh):
                    break
            gamma[d, :] = gammad
            # Contribution of document d to the expected sufficient
            # statistics for the M step.
            sstats[:, ids] += np.outer(expElogthetad.T, cts/phinorm)

        sstats = sstats * self._expElogbeta

        return((gamma, sstats))

def dirichlet_expectation(alpha):
    if (len(alpha.shape) == 1):
        return(psi(alpha) - psi(np.sum(alpha)))
    return(psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis])

def parse_doc_list(corpus, vocabulary):
    wordids = list()
    wordcts = list()
    for d in corpus:
        ddict = dict()
        for word in corpus[d]:
            wordtoken = vocabulary[word]
            ddict[wordtoken] = corpus[d][word]
        wordids.append(ddict.keys())
        wordcts.append(ddict.values())
    return ((wordids, wordcts))

def create_vocabulary(corpus):
    vocabulary = {}
    current_position = 0
    for document in corpus:
        for word in corpus[document]:
            if word not in vocabulary:
                vocabulary[word] = current_position
                current_position += 1
    return vocabulary

def main():
    corpus_series = pd.read_pickle(HOME_PATH + '/Projects/mlinb/heavy_pickles/corpus.pickle')
    corpus = corpus_series.to_dict()
    K = 10
    v_lda = Variational_LDA(corpus, K, 1./K, 1./K, 1024., 0.7)
    for iteration in range(0, 100):
        gamma = v_lda.update_lambda_docs()
        print gamma



if __name__ == '__main__':
    main()
