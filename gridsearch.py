import subprocess
import os
from typing import Any
import time
import random
from hparams_segmenter import config
from train_code_segmenter import train_tfidf, train_fasttext
from sklearn.model_selection import GridSearchCV
import warnings
import itertools

warnings.filterwarnings("ignore")   

def makeGrid(pars_dict):
    keys=pars_dict.keys()
    combinations=itertools.product(*pars_dict.values())
    print(combinations)
    ds=[dict(zip(keys,cc)) for cc in combinations]
    return ds


def generate_configs_tfidf():
    config = dict(
        data_dir="data",
        do_preprocessing=True,
        concat_strings=False,
        github_codebase_size=0.2,
        tfidf=True,
        num_of_previous_lines=3,
        num_of_previous_statistics=5,
        tfidf_config={
            'lowercase': [False],
            'analyzer': ['char'],
            'max_features': [1000 * i for i in range(1, 4, 2)],
            'ngram_range': [(1, 5)],
            'sublinear_tf': [True], 
        },
        model='LOGREG',
        model_config={
            'solver': 'lbfgs',
            'penalty': 'l2',
            'fit_intercept': True
        }
    )
    tfidf_grid = makeGrid(config['tfidf_config'])

    for github_codebase_size in [0.4 * i for i in range(3,10,5)]:
        for model in ['LOGREG']:
            for tfidf_config in tfidf_grid:
                config['tfidf_config'] = tfidf_config
                config['model'] = model
                config['github_codebase_size'] = github_codebase_size
                yield config

def tfidf_grid_search():
    for config in generate_configs_tfidf():
        begin = time.time()
        print(config['tfidf_config'])
        train_tfidf(config)
        print(f"FINISHED in {int(time.time() - begin)}s")

def generate_configs_fasttext():
    config = dict(
        data_dir=["data"],
        do_preprocessing=[True],
        tfidf=[False],
        github_codebase_size=[0.5 * i for i in range(10)],
        num_of_previous_lines=[1, 2, 3, 4],
        model=['fasttext']
    )
    fasttext_grid = makeGrid(config)
    for fasttext_config in fasttext_grid:
        yield fasttext_config

def fasttext_grid_search():
    for config in generate_configs_fasttext():
        begin = time.time()
        print(config)
        train_fasttext(config)
        print(f"FINISHED in {int(time.time() - begin)}s")
    
def train_custom_config():
    begin = time.time()
    config = dict(
        data_dir="data",
        do_preprocessing=True,
        tfidf=False,
        github_codebase_size=0.4,
        num_of_previous_lines=2,
        model='fasttext'
    )
    train_fasttext(config)
    print(f"FINISHED in {int(time.time() - begin)}s")


if __name__ == '__main__':
    tfidf_grid_search()