tags_to_save = ['javascript',
                'программирование',
                'разработка',
                'web-разработка',
                'веб-разработка',
                'ml',
                'ai']


config = dict(
    data_dir="data",
    tags_to_save=tags_to_save,
    train_size=200000,
    class_ratio=0.5,
    model="cointegrated/rubert-tiny2",
    tokenizer="cointegrated/rubert-tiny2",
    batch_size=32,
    is_finetuning=True,
    learning_rate=1e-5,
    epochs=4,
    num_warmup_steps=0,
)
