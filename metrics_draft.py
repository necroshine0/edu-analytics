import numpy as np


def euclidean(x, y):
    return (x - y) ** 2


def get_inds(labels, k):
    return np.argwhere(labels == k).reshape((-1,))


# cohesion (WSS with rho = euclidean) --> min
def cohesion(X, labels, rho=euclidean, centers=None, K=None):
    if K is None:
        K = len(np.unique(labels))

    def foo(k):
        inds = get_inds(labels, k)
        sub_X = X[inds]
        if centers is None:
            center = np.mean(sub_X, axis=0)
        else:
            center = centers[k]

        return rho(sub_X, center)

    return np.sum(list(map(foo, range(K))))


# separation (BSS with rho = euclidean) --> max
def separation(X, labels, rho=euclidean, centers=None, K=None):
    if K is None:
        K = len(np.unique(labels))

    mean = np.mean(X, axis=0)

    def foo(k):
        inds = get_inds(labels, k)
        sub_X = X[inds]
        if centers is None:
            center = np.mean(sub_X, axis=0)
        else:
            center = centers[k]

        return len(inds) * rho(mean, center) ** 2

    return np.sum(list(map(foo, range(K))))


# =====================================================Dunn Index======================================

def delta(X_1, X_2, variation='base'):
    if variation == 'base':
        _ = np.linalg.norm(np.repeat(X_1, len(X_2), axis=0) - np.array(list(X_2) * len(X_1)), axis=-1)
        delta = np.min(_)
    elif variation[-2] == '3':
        c = 1 / (len(X_1) * len(X_2))
        _ = np.linalg.norm(np.repeat(X_1, len(X_2), axis=0) - np.array(list(X_2) * len(X_1)), axis=-1)
        delta = c * np.sum(_)
    elif variation[-2] == '4':  # FIXME
        delta = np.linalg.norm(np.mean(X_1, axis=0) - np.mean(X_2, axis=0))
    elif variation[-2] == '5':
        c = 1 / (len(X_1) + len(X_2))
        mean_1 = np.mean(X_1, axis=0)
        mean_2 = np.mean(X_2, axis=0)
        sum_1 = np.sum(np.linalg.norm(X_1 - mean_1, axis=-1))
        sum_2 = np.sum(np.linalg.norm(X_2 - mean_2, axis=-1))
        delta = c * (sum_1 + sum_2)
    else:
        raise ValueError('Unknown method: wanted one from \
            base, gD31, gD41, gD51, gD33, gD43, gD53, given {}'.format(variation))

    return delta


def diam(X, variation='base'):
    if variation not in ['base', 'gD31', 'gD41', 'gD51', 'gD33', 'gD43', 'gD53']:
        raise ValueError('Unknown method: wanted one from \
            base, gD31, gD41, gD51, gD33, gD43, gD53, given {}'.format(variation))

    if len(X) == 0:
        raise ValueError('X contains no data')
    elif len(X) == 1:
        return 0.0

    if variation == 'base' or variation[-1] == '1':  # FIXME
        # _ = np.linalg.norm( np.repeat(X, len(X), axis=0) - np.array(list(X) * len(X)), axis=-1 )
        norms = []
        for i in range(len(X)):
            for j in range(i + 1, len(X)):
                norms += [np.linalg.norm(X[i, :] - X[j, :])]

        diameter = np.max(norms)
    elif variation[-1] == '3':
        c = 2 / len(X)
        mean = np.mean(X, axis=0)
        diameter = c * np.sum(np.linalg.norm(X - mean, axis=-1))

    return diameter


# Generalizes Dunn Index -0> max
def DunnIndex(X, labels, rho=euclidean, K=None, variation='base'):
    '''
        variation in {base, gD31, gD41, gD51, gD33, gD43, gD53}
    '''

    if K is None:
        K = len(np.unique(labels))

    min_deltas = []
    for i in range(K):
        deltas = []
        inds_i = get_inds(labels, i)
        X_i = X[inds_i]
        for j in range(i + 1, K):
            inds_j = get_inds(labels, j)
            X_j = X[inds_j]
            deltas += [delta(X_i, X_j, variation=variation)]
        if len(deltas) != 0:
            min_deltas += [np.min(deltas)]
    min_delta = np.min(min_deltas)

    diams = []
    for k in range(K):
        # print(k)
        inds_k = get_inds(labels, k)
        X_k = X[inds_k]
        diams += [diam(X_k, variation=variation)]
    max_diam = np.max(diams)

    return min_delta / max_diam


# ==================================================COP Index===================================================


def COPIndex(X, labels, rho=euclidean, K=None):
    if K is None:
        K = len(np.unique(labels))

    for k in range(K):
        inds = get_inds(labels, k)
        X_sub = X[inds]
        center = np.mean(X_sub, axis=0)
        numerator = np.sum(np.linalg.norm(X_sub - center, axis=1))

    return min_delta / max_diam


# ======================================== DUNN 41 ======================================


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
