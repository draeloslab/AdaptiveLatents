# TODO: add preprocessing functions to proSVD?
#       demeaning, normalizing, filters, etc.
#       add 2 function inputs: data preprocessing functions and post update functions

import numpy as np
from scipy.linalg import rq


class proSVD:
    # init attributes
    # k            int     - reduced dimension
    # w_len        int     - window length
    # w_shift      int     - how many cols does the window shift by?
    # decay_alpha  float   - forgetting parameter (no memory = 0 < alpha <= 1 = no forgetting)
    # trueSVD      bool    - whether or not basis should be rotated to true SVD basis (stored as attribute U)
    # history      int     - 0 indicates no history will be kept,
    #                       >0 indicates how many bases/singular values to keep
    # track_prev   bool    - whether you want to keep the previous Q (incurs copying - better way to do this?)
    def __init__(self, k, w_len=1, w_shift=None, decay_alpha=1, trueSVD=False, history=0, track_prev=True, mode='pro'):
        self.k = k
        self.w_len = w_len
        self.w_shift = w_len if w_shift is None else w_shift  # defauls to nonoverlapping chunks of w_len cols
        self.decay_alpha = decay_alpha
        self.trueSVD = trueSVD
        self.history = history
        self.track_prev = track_prev
        self.proj_mean = np.zeros((k))  # for global mean of projected data (to get projected variance)
        self.mode = mode

    def initialize(self, A_init, Q_init=None, B_init=None):
        ## Ainit just for initialization, so l1 is A.shape[1]
        n, l1 = A_init.shape

        ## make sure A_init.shape[1] >= k
        assert l1 >= self.k, "please init with # of cols >= k"

        self.global_mean = A_init.mean(axis=1)  # for global mean of observed data (for demeaning before projecting)

        # initialize Q and B from QR of A_init, W as I
        Q, B = np.linalg.qr(A_init, mode='reduced')
        ## TODO: other init strategies?
        self.Q = Q[:, :self.k] if Q_init is None else Q_init
        self.B = B[:self.k, :l1] if B_init is None else B_init

        if self.trueSVD:
            U_init, S_init, _ = np.linalg.svd(A_init, full_matrices=False)
            self.U = U_init[:, :self.k]
            self.S = S_init[:self.k]

        if self.history:
            ## these may need to be some kind of circular buffer
            ## for now assuming self.history = A_full.shape[1] - l1 (if not 0)
            self.Qs = np.zeros((n, self.k, self.history))
            self.Qs[:, :, 0] = self.Q  # init with first Q

            # keeping true singular vectors/values
            if self.trueSVD:
                self.Us = np.zeros(self.Qs.shape)
                self.Us[:, :, 0] = self.U
                self.Ss = np.zeros((self.k, self.history + 1))
                self.Ss[:, 0] = self.S
        self.t = 1

        # method to do common run through data (replaces pro.updateSVD())

    # make sure to pro.initialize() first!
    # inputs: data A, number of observations to init with
    #        optional num_iters, defaults to self.w_shift going through once
    # updates proSVD
    # outputs: projections, variance explained, derivatives
    def run(self, A, update_times, ref_basis=None):

        # run proSVD online
        for i, t in enumerate(update_times):
            start, end = t, t + self.w_len
            dat = A[:, start:end]

            # TODO: run should take user input pre/postupdate functions
            # they should be executed here
            self.preupdate()
            self.updateSVD(dat, ref_basis)
            self.postupdate()

        return

    # internal func to do a single iter of basis update given some data A
    def updateSVD(self, A, ref_basis=None):
        ## Update our basis vectors based on a chunk of new data
        ## Currently assume we get chunks as specificed in self.l
        ## QR decomposition of new data
        C = self.Q.T @ A
        A_perp = A - self.Q @ C
        Q_perp, B_perp = np.linalg.qr(A_perp, mode='reduced')

        # Calculate QR decomposition of augmented data matrix, Q_hat, R_hat
        # Q_hat is simple appending of Qi-1 and Q_perp
        Q_hat = np.concatenate((self.Q, Q_perp), axis=1)

        # R_hat is based on Figure 3.1 in Baker's thesis
        B_prev = np.concatenate((self.B, C), axis=1)
        tmp = np.zeros((B_perp.shape[0], self.B.shape[1]))
        tmp = np.concatenate((tmp, B_perp), axis=1)
        B_hat = np.concatenate((B_prev, tmp), axis=0)

        ## Constructing orthogonal Gu and Gv from Tu and Tv
        # SVD of B_hat
        U, diag, V = np.linalg.svd(B_hat, full_matrices=False)

        # decaying singular values (implements forgetting)
        diag *= self.decay_alpha

        # getting Tu
        if self.mode == 'pro':  # Orthogonal Procrustes solution
            if ref_basis is not None:  # solution for a 'reference' basis
                Mu = ref_basis.T @ Q_hat @ U[:, :self.k]
            else:  # solution for minimum change from previous basis
                Mu = U[:self.k, :self.k]  # same as Mu = self.Q.T @ Q_hat @ U[:, :self.k]

            U_tilda, _, V_tilda_T = np.linalg.svd(Mu, full_matrices=False)
            Tu = U_tilda @ V_tilda_T
            Gu_1 = U[:, :self.k] @ Tu.T
        else:
            # Gu_1, Tu = rq(U[:, :self.k])  # Baker et al.
            Gu_1 = U[:, :self.k]  # Sequential KL
            Tu = Gu_1.T @ U[:, :self.k]

        # getting Tv
        if True:  # self.Tv_mode == 'baker':
            Gv_1, Tv = rq(V[:, :self.k])
        else:
            Gv_1 = V[:, :self.k]  # sequential KLT
            Tv = Gv_1.T @ V[:, :self.k]

        ## UPDATING Q, B
        self.Q = Q_hat @ Gu_1
        self.B = Tu @ np.diag(diag[:self.k]) @ Tv.T

        # Getting true SVD basis
        if self.mode != 'seq-kl':
            if self.trueSVD:
                U, S, V = np.linalg.svd(self.B)
                self.U = self.Q @ U
                self.S = S

        return

    # preupdate
    def preupdate(self):
        # make copy to keep previous basis
        if self.track_prev:
            self.Q_prev = np.copy(self.Q)

    # postupdate
    def postupdate(self):
        # updating history
        if self.history:
            self.Qs[:, :, self.t] = self.Q
            if self.trueSVD:
                self.Us[:, :, self.t] = self.U
                self.Ss[:, self.t] = self.S

        self.t += 1