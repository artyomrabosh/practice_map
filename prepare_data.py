from hparams import config
import pandas as pd
import os
from utils.parsers import Parser
from tqdm import tqdm
from utils.datasets import LabeledDataset


def custom_str_to_lst(string):
    return string[1:-1].replace("'", "").split(", ")


def prepare_data_habr():
    df = pd.read_csv(os.path.join(config['data_dir'], 'habr', "raw_texts.csv"))
    df['tags'] = df.tags.apply(custom_str_to_lst)
    mask = df.tags.apply(lambda x: bool(list(set(x) & set(config['tags_to_save']))))
    res = df[mask]
    res["text"] = res['text'].apply(str)

    parser = Parser()

    sents = []

    for text in tqdm(res.text):
        sents += parser.text_to_sents(text)
        if len(sents) > config['train_size']:
            break

    df = pd.DataFrame({"text": sents})
    df.to_csv(os.path.join(config['data_dir'], 'sents', "habr.csv"), encoding='utf-8')


def prepare_data_spbu():
    assert os.path.isfile(os.path.join(config['data_dir'], 'sents', 'spbu.csv'))


def prepare_data():
    prepare_data_spbu()
    prepare_data_habr()
    dataset = LabeledDataset(config['class_ratio'])

    dataset.save_train(config['train_size'])
