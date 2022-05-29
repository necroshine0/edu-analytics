import matplotlib.pyplot as plt
from matplotlib.colors import Colormap, ListedColormap
import seaborn as sns

sns.set(style='whitegrid')
palette = sns.color_palette('tab10', n_colors=50)
cmap = ListedColormap(palette)

colors = {
    0: 'teal',
    1: 'darkturquoise',
    2: 'red',
    3: 'crimson',
    4: 'darkorange',
    5: 'gold',
    6: 'magenta',
    7: 'purple',
    8: 'darkorchid',
    9: 'lime',
    10: 'limegreen',
    11: 'mediumvioletred',
    12: 'sienna',
    13: 'limegreen',
    14: 'royalblue',
    15: 'mediumblue',
}

import pandas as pd
import numpy as np
from wordcloud import WordCloud

def draw_wordcloud(texts, max_words=1000, width=900, height=400, random_state=10):
    wordcloud = WordCloud(background_color='white', max_words=max_words,
                          width=width, height=height, random_state=random_state)

    joint_texts = ' '.join(list(texts))
    wordcloud.generate(joint_texts)
    return wordcloud.to_image()


def draw_cluster_clouds(data, clusters, n_clusters, alert_by='text', cloud_kwargs={}):
    for i in range(n_clusters):
        print('cluster:', i + 1)
        inds = np.argwhere(clusters == i)
        if len(inds) == 0:
            print('empty')
            continue
        display(draw_wordcloud(data.iloc[inds.reshape((-1,))][alert_by], **cloud_kwargs))


def plot_data_embs(data_tsne: dict, title=''):
    fig, axes = plt.subplots(nrows=1, ncols=len(data_tsne), figsize=(24, 6))
    for j, (emb_name, X_emb) in enumerate(data_tsne.items()):
        axes[j].scatter(X_emb[:, 0], X_emb[:, 1], linewidths=1, c='royalblue', alpha=.8)
        axes[j].set_title(emb_name, fontsize=15)
        if j == 0:
            axes[j].set_ylabel('dim 2', fontsize=15)
        axes[j].set_xlabel('dim 1', fontsize=15)

    if title != '':
        fig.suptitle(title, fontsize=20)
    plt.show()


def get_alpha(frac):
    '''
        dynamic color transparency function
    '''
    return 0.7 * max(0.1, np.tanh(1.0 - 1.4 * frac) / 0.8)


def plot_clustering(n_clusters, data_tsne: dict, labels: dict, methods: list, embeddings: list, cmap=cmap, title='Clustering results'):
    '''
        n_clusters: int -- number of clusters
        data_tsne: dict -- fitted t-SNE embeddings for all embeddings given
        methods: list of str -- clustering algs. names
        embeddings: list of str -- embeddings names
        labels: list of dicts -- clustering labels for all agls. and emb-s

    '''
    assert len(methods) == len(labels)

    figsize = (5 * len(methods), 5 * len(data_tsne))
    fig, axes = plt.subplots(nrows=len(embeddings), ncols=len(methods), figsize=figsize, squeeze=False)

    for j, emb_name in enumerate(embeddings):
        X_emb = data_tsne[emb_name].copy()
        for i, method_name in enumerate(methods):
            labels_method = labels[i][emb_name]
            L = len(labels_method)
            for k in range(n_clusters):
                inds = np.argwhere(labels_method == k).reshape((-1,))

                axes[j][i].scatter(X_emb[inds, 0], X_emb[inds, 1], c=colors[k],
                                   linewidths=0.7, alpha=get_alpha(len(inds) / L))

            if j == 0:
                axes[j][i].set_title(method_name, fontsize=15)
            if i == 0:
                axes[j][i].set_ylabel(emb_name, fontsize=15)

    if title != '':
        fig.suptitle(title, fontsize=20)
    # plt.legend()
    plt.show()
