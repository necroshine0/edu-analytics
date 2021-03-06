import numpy as np

def get_inds(labels, k):
    return np.argwhere(labels == k).reshape((-1,))


# ======================================= Within-group scatter ====================================

# Within-Group Matrix for cluster k
def WGk(X, labels=None, k=None, precomputed=False, return_trace=True):
    '''
        if precomputed:
            X -- matrix formed by the centered columns vectors v_j - u_j of cluster k
        else:
            X -- data embedding
    '''
    if precomputed:
        X_k = X.copy()
    else:
        if labels is None or k is None:
            raise ValueError('labels and k variables can\'t be None if precomputed is True')

        inds_k = get_inds(labels, k)
        A_k = X[inds_k].copy()
        centers = np.mean(A_k, axis=0)
        X_k = A_k - centers

    if not return_trace:
        return X_k.T @ X_k
    else:
        diag = np.sum(X_k ** 2, axis=0)
        return np.sum(diag)


# Within-Cluster Dispersion for all clusters -- trace of all WGk
def WGSS(X, labels, K=None):
    if K is None:
        K = len(np.unique(labels))

    wgss = 0.0
    for k in range(K):
        wgss += WGk(X, labels, k)
    return wgss


# standard between cluster distance
def between_clust_dist(X_1, X_2, type='1'):
    if type not in ['1']:
        raise ValueError('invalid distance type')

    deltas = []
    for i in range(len(X_1)):
        for j in range(len(X_2)):
            deltas += [ np.linalg.norm(X_1[i] - X_2[j]) ]
    return np.min(deltas)

# ======================================== Xie-Beni ======================================

# Xie-Beni Index --> min
def XieBeniIndex(X, labels, K=None):
    if K is None:
        K = len(np.unique(labels))

    min_deltas = []
    for i in range(K):
        inds_i = get_inds(labels, i)
        X_i = X[inds_i].copy()
        for j in range(i + 1, K):
            inds_j = get_inds(labels, j)
            X_j = X[inds_j].copy()
            min_deltas += [ between_clust_dist(X_i, X_j) ]

    delta_btw = np.min(min_deltas)
    wgss = WGSS(X, labels, K)

    return wgss / delta_btw / len(X)


# ======================================= B-Cubed =====================================

class BCubed(object):
    def correctness(self, y, ax):
        '''
            computes correctness matrix:
            C(ij) = [yi == yj] * [a(xi) == a(xj)]
            y  - labels
            ax - predictions
        '''
        assert y.shape == ax.shape and len(y.shape) == 1
        matching_y = (y.reshape((-1, 1)) == y).astype(int)
        matching_ax = (ax.reshape((-1, 1)) == ax).astype(int)
        # C = matching_y * matching_ax
        C_sum = (matching_y * matching_ax).sum(axis=1)
        return C_sum, matching_y.sum(axis=1), matching_ax.sum(axis=1)


    def precision(self, C_sum, ax_sum):
        return np.mean(C_sum / ax_sum)

    def recall(self, C_sum, y_sum):
        return np.mean(C_sum / y_sum)


    def fscore(self, y, ax, beta=None):
        C_sum, y_sum, ax_sum = self.correctness(y, ax)
        prec, rec = self.precision(C_sum, ax_sum), self.recall(C_sum, y_sum)

        if beta is None:
            fscore = (2 * prec * rec) / (prec + rec)
        else:
            fscore = (1.0 + beta ** 2) * (prec * rec / (beta ** 2 * prec + rec))
        return fscore