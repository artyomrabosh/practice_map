config = dict(
    data_dir="data",
    do_preprocessing=True,
    github_codebase_size=0.2,
    num_of_previous_lines=3,
    num_of_previous_statistics=5,
    tfidf=True,
    tfidf_config={
        'lowercase':False,
        'analyzer': 'word',
        'max_features': 20000
    },
    model='LOGREG',
    model_config={
        'solver': 'lbfgs',
        'penalty': 'l2',
        'fit_intercept': True
    }
)

