import multiprocessing
cores = multiprocessing.cpu_count()

lda_kwargs = {
    'learning_method':'online',
    'batch_size':128,
    'learning_decay':0.9,
    'n_components':16,
    'random_state':10,
}

tfidf_kwargs = {
    'encoding':'ansi',
    'decode_error':'ignore',
    'strip_accents':'unicode',
    'analyzer':'word',
    'ngram_range':(1, 1),
}

w2v_kwargs = {
    'vector_size':1500,
    'window':10,
    'workers':cores - 1,
    'epochs':15,
    'negative':9,
}

d2v_kwargs = {
    'vector_size':1500,
    'window':10,
    'workers':cores - 1,
    'epochs':15,
    'negative':9,
}

tsne_kwargs = {
    'metric':'cosine',
    'learning_rate':'auto',
    'min_grad_norm':1e-8,
    'random_state':10,
    'init':'pca',
    'perplexity':20,
    'method':'barnes_hut',
}

# pca_kwargs = {
#     'n_components':2,
#     'random_state':30,
#     'tol':1e-6,
#     'svd_solver':'full',
# }

kmeans_kwargs = [
    *[{
        'n_clusters':16,
        'init':'k-means++',
        'tol':1e-7,
        'max_iter':200,
        'random_state':10,
        'algorithm':'elkan',
    }] * 3,

    {
        'n_clusters':16,
        'init':'random',
        'tol':1e-3,
        'max_iter':150,
        'random_state':10,
        'algorithm':'elkan',
    }
]

gm_kwargs = [
    *[{
        'n_components':16,
        'covariance_type':'diag',
        'tol':1e-5,
        'reg_covar':1e-2,
        'max_iter':200,
        'n_init':5,
        'init_params':'kmeans',
        'random_state':10,
    }] * 3,

    {
        'n_components':16,
        'covariance_type':'spherical',
        'tol':1e-5,
        'reg_covar':1e-3,
        'max_iter':300,
        'n_init':7,
        'init_params':'kmeans',
        'random_state':10,
    }
]

spectral_kwargs = [
    {
        'n_clusters':16,
        'n_components':40,
        'assign_labels':'kmeans',
        'n_init':7,
        'eigen_solver':'lobpcg',
        'random_state':10,
        'affinity':'nearest_neighbors',
        'n_neighbors':12,
        'n_jobs':-1,
    },

    {
        'n_clusters':16,
        'n_components':120,
        'assign_labels':'kmeans',
        'n_init':10,
        'eigen_solver':'lobpcg',
        'random_state':10,
        'affinity':'nearest_neighbors',
        'n_neighbors':15,
        'n_jobs':-1,
    },

    {
        'n_clusters':16,
        'n_components':100,
        'assign_labels':'kmeans',
        'n_init':5,
        'eigen_solver':'arpack',
        'random_state':10,
        'affinity':'nearest_neighbors',
        'n_neighbors':10,
        'n_jobs':-1,
    },

    {
        'n_clusters':16,
        'n_components':20,
        'assign_labels':'kmeans',
        'n_init':4,
        'eigen_solver':'lobpcg', # lobpcg/arpack
        'random_state':10,
        'affinity':'nearest_neighbors',
        'n_neighbors':40,
        'n_jobs':-1,
    },
]