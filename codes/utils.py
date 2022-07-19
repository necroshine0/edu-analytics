import pandas as pd
import numpy as np
import os
import re
import glob
from time import sleep

from tqdm.notebook import tqdm
from IPython.display import display, clear_output

from collections import defaultdict
from scipy.sparse.csr import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from hyperparameters import tfidf_kwargs

from visualization import get_inds

    
class idf_mean_vectorizer(object):
    def __init__(self, word2vec_dict):
        self.word2vec = word2vec_dict
        self.word2weight = None
        self.dim = len(next(iter(word2vec_dict.values())))

    def fit(self, corpus: list):
        tfidf = TfidfVectorizer(**tfidf_kwargs).fit(corpus)
        max_idf = np.max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, corpus):
        corpus_splitted = [text.split() for text in corpus]
        return csr_matrix([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in corpus_splitted
            ])
    
    def fit_transform(self, corpus: list):
        self = self.fit(corpus)
        return self.transform(corpus)
        
        
def reclusterize(row, by, new_cluster: str):
    if type(by) == str:
        if by in row['text'] and by in row['name']:
            return new_cluster
    elif type(by) == list:
        for elem in by:
            if elem in row['text'] and elem in row['name']:
                return new_cluster
    
    return row['cluster']


def markup(df, labels, label, inds=None, folder='markup'):
    '''
        label > 0
    '''
    try:
        os.makedirs(folder)
    except:
        pass

    if inds is None:
        inds = get_inds(labels, label - 1)
    df_tmp = df.iloc[inds].copy()

    N = len(df_tmp)
    for num, ind in enumerate(inds):
        clear_output(wait=True)
        print('{} from {}'.format(num, N - 1))

        r = df_tmp.loc[ind]
        print(r['name'].upper(), '\n\t', r['cluster'])
        sleep(0.35)

        _ = input()
        if _ == 'text':
            display(r['text'])
            sleep(0.35)
            _ = input()

        if _ == 'break':
            file.close()
            return
        elif _ == '' or _ == '\n':
            _ = r['cluster']

        try:
            file = open(folder + '\\{}.txt'.format(label), 'a', encoding='ansi')
            file.write(_ + '; ')
            file.close()
        except Exception as e:
            print('FATAL:', e)
            break