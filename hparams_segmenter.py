config = dict(
    data_dir="data",
    do_preprocessing=True,
    github_codebase_size=0.2,
    num_of_previous_lines=4,
    tfidf=True,
    tfidf_config={
        'lowercase':False,
        'analyzer': 'word',
        'max_features': 20000
    }
)
