import torch

from IPython.display import clear_output
import pandas as pd
import os
from os.path import isdir, isfile
from natasha import (
    Segmenter,

    NewsEmbedding,
    NewsMorphTagger,

    Doc
)

class PDFDataset(torch.utils.data.Dataset):
    
    def check_pdfs(self):
        self.pdfs = []

        assert isdir(self.data_dir) == True
        
        for path in os.listdir(self.data_dir):
            if os.path.isfile(os.path.join(self.data_dir, path)):
                self.pdfs.append(path)
    
    def __init__(self, data_dir='data'):
        self.data_dir = os.path.join(data_dir, 'spbu', "pdf")
        self.check_pdfs()
    
    def __getitem__(self, idx):
        return {'name': self.pdfs[idx], "path": os.path.join(self.data_dir, self.pdfs[idx])} 
    
    def __len__(self):
        return len(self.pdfs)
    
    def read_pdf(self, idx):
        return open(os.path.join(self.data_dir, self.pdfs[idx]), 'rb')

class SentsDataset(torch.utils.data.Dataset):

    @staticmethod
    def read_df(path):
        df = pd.read_csv(path)
        source_name = os.path.basename(path[:-4])
        return pd.DataFrame({"source_name": len(df) * [source_name],
                             "text": df.text})

    @staticmethod
    def is_sing_first_person(doc):
        for token in doc.morph.tokens:
            if "Number" in token.feats and "Person" in token.feats:
                if token.feats['Person'] == "1" and token.feats["Number"] == "Sing":
                    return True
        return False

    def __init__(self, data_dir="data", create_if_exist=False):
        self.data_dir = os.path.join(data_dir, "sents")
        assert isdir(os.path.join('data', "sents")) == True

        dataframes = []

        if "sents.csv" in os.listdir(self.data_dir):
            self.df = pd.read_csv(os.path.join(self.data_dir, "sents.csv"))
            return

        for path in os.listdir(self.data_dir):
            if isfile(os.path.join(self.data_dir, path)) and not path.startswith("labeled"):
                dataframes.append(self.read_df(os.path.join(self.data_dir, path)))

        self.df = pd.concat(dataframes)
        self.df = self.df.sample(n=20000)
        self.analyze_morphology()

    def analyze_morphology(self):
        emb = NewsEmbedding()
        morph_tagger = NewsMorphTagger(emb)
        segmenter = Segmenter()

        self.df['doc'] = self.df.text.progress_apply(Doc)
        self.df.doc.progress_apply(lambda x: x.segment(segmenter))
        self.df.doc.progress_apply(lambda x: x.tag_morph(morph_tagger))
        self.df['is_sing'] = self.df.doc.progress_apply(self.is_sing_first_person)

    def save(self):
        self.df.to_csv(os.path.join(self.data_dir, "sents.csv"))

    def save_train(self):
        labeling = {
            "habr": 0,
            "spbu": 1
        }
        df_to_save = self.df

        self.df['label'] = self.df.source_name.apply(lambda x: labeling[x])

        self.df.to_csv(os.path.join(self.data_dir, "train.csv"))

    def sample(self, n=1):
        if n == 1:
            return self.df.sample()
        return self.df.sample(n)

    def __getitem__(self, i):
        return self.df[i]

    def __len__(self):
        return len(self.df)

    def __contains__(self, text):
        return text in list(self.df.text)

class LabeledDataset(SentsDataset):
    
    LABELS = {
        'y' : 1, # scientific
        'n' : 0  # non scientific
    }
    
    def __init__(self, data_dir="data", create_if_exist=False):
        super().__init__(data_dir, create_if_exist)
        
        if not os.path.isfile(os.path.join(f"{self.data_dir}", "labeled.csv")):
            self.labeled_df = pd.DataFrame({'text': [], "source_name": [], 'label': []})
        else:
            self.labeled_df = pd.read_csv(os.path.join(f"{self.data_dir}", "labeled.csv"))
        
    def save(self):
        self.df.to_csv(os.path.join(f"{self.data_dir}", "sents.csv"), index=False)
    
    def save_labeled(self):
        self.labeled_df.to_csv(os.path.join(f"{self.data_dir}", "labeled.csv"), index=False)
    
    def add_new_record(self, record, label):
        new_record = {'text': record.text.item(),
                      'source_name': record.source_name.item(),
                      'label': label}
        self.labeled_df = self.labeled_df.append(new_record, ignore_index=True)
        self.save_labeled()
    
    def label_df(self):
        inp = "s"
        while inp != 'exit':
            record = self.df.sample()
            
            if record.text.item() in self.labeled_df.text:
                continue

            clear_output(wait=False)
            print(f'Source  : {record.source_name.item()}')
            print(f'Text    : {record.text.item()}')
            
            inp = input().lower()
            while inp not in ["y", "n", 's', 'exit']:
                print("invalid command")
                print("valid commands: Y, N, S, EXIT")
                inp = input().lower()
            
            if inp in ['y', 'n']:
                self.add_new_record(record, self.LABELS[inp])
    
            if inp == "exit":
                self.save_labeled()