import wandb
import datetime
from hparams_segmenter import config
from prepare_data import prepare_data_segmenter, prepare_data_fasttext, agg_func
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem.snowball import PorterStemmer, SnowballStemmer

from segmenter_features import *

from scipy.sparse import hstack
import fasttext
import pandas as pd
import numpy as np
import warnings


def start_wandb(config=config):
    name = 'test-({})'.format(
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    wandb_config = config.copy()

    wandb_run = wandb.init(project='code_segmenter', name=name, config=wandb_config)
    wandb_run.log_code('.')
    return wandb_run

def log_report(report, sample):
    wandb_report = {}
    for label in report:
        if type(report[label]) == float:
            wandb_report[sample + '_' + label] = report[label]
        else:
            for metric in report[label]:
                wandb_report[sample + '_' + label+'_' + metric] = report[label][metric]
    wandb.log(wandb_report)

def train_conv(config=config):
    raise NotImplementedError

def preprocessing_str(s):
    s = re.sub('[0-9]+', ' NUMBER ', s)
    s = s.replace('\t', '')
    s = s.replace('\n', '')
    s = s.replace('\r', '')
    s = str(s)
    s = str(s).replace(',',' , ').replace('"','').replace('\'',' \' ').replace('.',' . ').replace('(',' ( ').\
            replace(')',' ) ').replace('!',' ! ').replace('?',' ? ').replace(':',' : ').replace(';',' ; ').\
            replace('<', ' < ').replace('>', ' > ').replace('-', ' - ').replace('*', ' * ').lower()
    return s

def train_fasttext(config=config):
    start_wandb(config=config)
    if config['do_preprocessing']:  
        prepare_data_segmenter(config=config)
        prepare_data_fasttext(config=config)  
        
    print('Successfull preprocessing')
    model = fasttext.train_supervised('training_temp/train.txt', 
                                      autotuneValidationFile='training_temp/val.txt', 
                                      autotunePredictions='recallAtPrecision:50:__label__code')
    
    print('Successfull training')

    val = pd.read_csv('training_temp/val.csv')
    train = pd.read_csv('training_temp/train.csv')

    def predict(line):
        try:
            labels, probs = model.predict(line)
        except:
            return "text"
        label = labels[0][9:]
        return label
    
    val['predicted'] = val.line.apply(predict)
    train['predicted'] = train.line.apply(predict)
    
    log_report(classification_report(train['label'], train['predicted'], output_dict=True), 'train')
    log_report(classification_report(val['label'], val['predicted'], output_dict=True), 'val')

    with open('training_temp/val_result.txt', 'w') as f:
        print("STARTED LABELING")
        for text, predicted, label in zip(val['line'], val['predicted'], val['label']):
            f.writelines(f'__label__{label.upper()}__{predicted.upper()} {text}\n')

    print('Successfull metrics')
    with open("run_id.txt", "w+") as f:
        print(wandb.run.id, file=f)
    wandb.run.finish()

def vectorize_text(train, val):
    stemmer_en = PorterStemmer()
    stemmer_ru = PorterStemmer('russian')
    analyzer = CountVectorizer().build_analyzer()

    def stemmed_words_punct(doc):
        words = re.findall(reg_word, doc)
        punct = re.findall('[^\w\s]', doc)
        words = [stemmer_ru.stem(stemmer_en.stem(w)) for w in analyzer(doc)]
        return punct + words
    
    print('start fit vect')

    # vectorizer = TfidfVectorizer(**config['tfidf_config'])

    vectorizer = CountVectorizer(lowercase=False,
                max_features= 1000,
                analyzer=stemmed_words_punct)

    vectorizer.fit(train.line.apply(lambda x: np.str_(x)))

    tfidf_train = vectorizer.transform(train.line.apply(lambda x: np.str_(x)))
    tfidf_train = tfidf_train.toarray()

    tfidf_val = vectorizer.transform(val.line.apply(lambda x: np.str_(x)))
    tfidf_val = tfidf_val.toarray()
    print('vect fitted')
    return tfidf_train, tfidf_val

def generate_features(train, val):
    features = [end_of_line, code_reg, is_link, is_camelcase, 
            num_of_words, len, is_comment, is_attribute, 
            num_of_index_letter, num_of_index_num, num_of_words_ru, num_of_words_en, brackets]
    
    features_train = []
    features_val = []

    for feature in features:
        col_train = train.feature_line.progress_apply(feature).to_numpy().reshape(len(train), 1)
        col_val = val.feature_line.apply(feature).to_numpy().reshape(len(val), 1)
        features_train.append(col_train)
        features_val.append(col_val)

    features_train = np.hstack(features_train)
    features_val = np.hstack(features_val)
    return features_train, features_val



def train_tfidf(config=config):
    start_wandb(config=config)
    prepare_data_segmenter(config=config)
    print('Successfull preprocessing')

    train = pd.read_csv('training_temp/train.csv')
    val = pd.read_csv("training_temp/val.csv")

    train = train.fillna(' ') 
    val = val.fillna(' ')
    
    train['line_num'] = (train.groupby(['work_id']).cumcount()+1)/train.groupby(["work_id"])["line"].transform("count")
    val['line_num'] = (val.groupby(['work_id']).cumcount()+1)/val.groupby(["work_id"])["line"].transform("count")

    train['feature_line'] = train.line
    val['feature_line'] = val.line

    train.line = train.line.apply(preprocessing_str)
    val.line = val.line.apply(preprocessing_str)    

    if config['do_preprocessing']:
        tfidf_train, tfidf_val = vectorize_text(train.line, val.line)
    else:
        with open('train_count_vect.npy', 'rb') as f:
            tfidf_train = np.load(f)
        with open('val_count_vect.npy', 'rb') as f:
            tfidf_val = np.load(f) 

    features_train, features_val = generate_features(train, val)

    
    rolled_features_train = [np.roll(features_train, shift=i, axis=0) for i in range(config['num_of_previous_statistics'])]
    rolled_features_val = [np.roll(features_val, shift=i, axis=0) for i in range(config['num_of_previous_statistics'])]
    
    X_train = [np.roll(tfidf_train, shift=i, axis=0) for i in range(config['num_of_previous_lines'])]
    X_val = [np.roll(tfidf_val, shift=i, axis=0) for i in range(config['num_of_previous_lines'])]

    X_train = np.hstack(X_train + rolled_features_train)
    X_val = np.hstack(X_val + rolled_features_val)


    match config['model']:
        case 'LOGREG':
            model = LogisticRegression(fit_intercept=config['model']['fit_intercept'], 
                            solver=config['model']['solver'], 
                            max_iter=100, 
                            penalty=config['model']['penalty'],
                            ).fit(X_train, y_train)
    
        case 'MLP':
            model = MLPClassifier(hidden_layer_sizes=(2,), 
                            solver='lbfgs',
                            verbose=False).fit(X_train, y_train)
        case 'CatBoost':
            X_train = pd.DataFrame(X_train)
            y_train = pd.DataFrame(y_train)
            model = CatBoostClassifier(iterations=config['model_config']['iterations'],
                                       depth=config['model_config']['dept'],
                                       l2_leaf_reg=config['model_config']['l2_leaf_reg']).fit(X_train, y_train)
    print('Successfull training')

    y_train_predicted = model.predict(X_train)
    y_val_predicted = model.predict(X_val)

    val_df = pd.DataFrame({"line": val.line, "label":val.label , "predicted":y_val_predicted})
    val_df.to_csv("training_temp/val_research.csv")

    log_report(classification_report(y_train, y_train_predicted, output_dict=True), 'train')
    log_report(classification_report(val_df.label, val_df.predicted, output_dict=True), 'val')
    print('Successfull metrics')

    with open("run_id.txt", "w+") as f:
        print(wandb.run.id, file=f)
    wandb.run.finish()


# if __name__ == '__main__':
#     train_nn()
