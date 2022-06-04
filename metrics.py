import numpy as np

# ======================================== DUNN ======================================


def get_inds(labels, k):
    return np.argwhere(labels == k).reshape((-1,))


def delta(X_1, X_2, variation='base'):

    if variation[-2] == '4':
        delta = np.linalg.norm(np.mean(X_1, axis=0) - np.mean(X_2, axis=0))
    else:
        raise ValueError('Unknown method: wanted one from \
            gD41, given {}'.format(variation))

    return delta


def diam(X, variation='base'):
    if variation not in ['base', 'gD31', 'gD41', 'gD51']:
        raise ValueError('Unknown method: wanted one from \
            base, gD31, gD41, gD51, given {}'.format(variation))

    if len(X) == 0:
        raise ValueError('X contains no data')
    elif len(X) == 1:
        return 0.0

    if variation == 'base' or variation[-1] == '1':
        # _ = np.linalg.norm( np.repeat(X, len(X), axis=0) - np.array(list(X) * len(X)), axis=-1 )
        norms = []
        for i in range(len(X)):
            for j in range(i + 1, len(X)):
                norms += [np.linalg.norm(X[i, :] - X[j, :])]

        diameter = np.max(norms)

    return diameter


# Generalizes Dunn Index gD41
def DunnIndex(X, labels, K=None, variation='base'):
    '''
        variation in {gD41}
    '''

    if K is None:
        K = len(np.unique(labels))

    min_deltas = []
    for i in range(K):  # ck
        deltas = []
        inds_i = get_inds(labels, i)
        X_i = X[inds_i].copy()
        for j in range(i + 1, K):  # cl
            inds_j = get_inds(labels, j)
            X_j = X[inds_j].copy()
            deltas += [delta(X_i, X_j, variation=variation)]
        if len(deltas) != 0:
            min_deltas += [np.min(deltas)]
    min_delta = np.min(min_deltas)

    diams = []
    for k in range(K):
        inds_k = get_inds(labels, k)
        X_k = X[inds_k].copy()
        diams += [diam(X_k, variation=variation)]
    max_diam = np.max(diams)

    return min_delta / max_diam


# ======================================== COP ======================================


# COP Index --> min
def COPIndex(X, labels, K=None):

    if K is None:
        K = len(np.unique(labels))

    def dists_matrix():
        M = np.zeros((len(X), len(X)))
        for i in range(len(X)):
            for j in range(i + 1, len(X)):
                M[i, j] = np.linalg.norm(X[i] - X[j])
                M[j, i] = M[i, j]
        return M

    M = dists_matrix()
    all_inds = np.arange(0, len(X))

    def get_dist(inds_k):
        sepped = np.setdiff1d(all_inds, inds_k)
        M_sliced = M[inds_k, :][:, sepped].copy()
        return np.min(np.max(M_sliced, axis=0))

    COP = 0.0
    for k in range(K):
        inds_k = get_inds(labels, k)
        X_sub = X[inds_k].copy()
        center = np.mean(X_sub, axis=0)
        numerator = np.sum(np.linalg.norm(X_sub - center, axis=1))
        denominator = get_dist(inds_k)
        COP += numerator / denominator
        
    return COP / len(X)
