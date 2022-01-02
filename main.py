from gensim import downloader
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
from sklearn import svm
from itertools import repeat
from sklearn.metrics import f1_score
from torch.optim import Adam
from model import WordsClassifier
from dataset import WordsDataSet
from train import train
from joblib import dump, load

import random
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

PRETRAINED_PATH = 'fasttext-wiki-news-subwords-300'
pretrained = downloader.load(PRETRAINED_PATH)


def load_file(file_name):
    with open(file_name) as f:
        lines = f.readlines()
    return lines


# Create df and list of sentences from loaded lines.
def lines_to_df(lines):
    df = pd.DataFrame(columns=['word', 'tag'])
    words_list = []
    tags_list = []
    sentences = []
    sentence = []
    for idx, item in enumerate(lines):
        # add word that represent start of sentence
        if idx == 0 or lines[idx - 1] == '\n':
            words_list.append('SOS')
            tags_list.append('O')
            sentence.append('SOS')
            continue
        # add word that represent end of sentence
        if lines[idx] == '\n':
            words_list.append('EOS')
            tags_list.append('O')
            sentence.append('EOS')
            sentences.append(sentence)
            sentence = []
            continue
        word = item.split()[0]
        tag = item.split()[1]
        words_list.append(word)
        sentence.append(word)
        if tag == 'O':
            tags_list.append(tag)
        else:
            tags_list.append('Y')
    df['word'] = words_list
    df['tag'] = tags_list
    return df, sentences


# apply Word2Vec model with given sentences
def vec_representation(sentences):
    model = Word2Vec(sentences=sentences, vector_size=300, window=3, min_count=1, workers=4, epochs=100)
    return model


# add column to df with the embedding vector
# we took our Word2Vec representation only for words that doesn't appear on pretrained model
def add_vector_col(df):
    df['vector'] = [[] for i in repeat(None, df.shape[0])]
    for index, row in df.iterrows():
        if row['word'] not in pretrained.key_to_index:
            df.loc[index, 'vector'] = list(word_vectors.vectors[word_vectors.key_to_index[row['word']]])
        else:
            df.loc[index, 'vector'] = list(pretrained[row['word']])
    return df


# concat embedding vectors for neighborhoods words that gave the context, one
def add_context(df):
    df['new_vector'] = [[] for i in repeat(None, df.shape[0])]
    for index, row in df.iterrows():
        if index == 0:
            window_vec = df['vector'][index] + df['vector'][index] + df['vector'][index + 1]
        elif index == len(df) - 1:
            window_vec = df['vector'][index - 1] + df['vector'][index] + df['vector'][index]
        else:
            window_vec = df['vector'][index - 1] + df['vector'][index] + df['vector'][index + 1]
        df['new_vector'][index] = window_vec
    return df


# run SVM model (model 1)
def simple_model(train_df, dev_df):
    lin_clf = svm.LinearSVC(loss='hinge', max_iter=100000)
    lin_clf.fit([row for row in train_df['new_vector']], train_df['tag'])
    lin_predicted_dev_tags = lin_clf.predict([row for row in dev_df['new_vector']])
    bi_f1_score_lin = f1_score(dev_df['tag'], lin_predicted_dev_tags, pos_label='Y')
    print(f"The F1 score for svm is: {bi_f1_score_lin:.4f}")
    dump(lin_clf, 'svm.joblib')
    return bi_f1_score_lin


# run neural networks model for model 2 and model 3
def NN_model(train_df, dev_df):
    train_ds = WordsDataSet(train_df)
    test_ds = WordsDataSet(dev_df)
    datasets = {"train": train_ds, "test": test_ds}
    model = WordsClassifier(num_classes=2, vocab_size=train_ds.vocabulary_size)
    optimizer = Adam(params=model.parameters())
    net, f1 = train(model=model, data_sets=datasets, optimizer=optimizer, num_epochs=10)
    print(f"The F1 score for NN model is: {f1:.4f}")
    torch.save(net.state_dict(), 'm2.pt')


if __name__ == '__main__':
    train_lines = load_file("train.tagged")
    train_df, train_sentences = lines_to_df(train_lines)
    dev_lines = load_file("dev.tagged")
    dev_df, dev_sentences = lines_to_df(dev_lines)

    sentences = train_sentences + dev_sentences

    model = vec_representation(sentences)
    word_vectors = model.wv
    word_vectors.save("word2vec.wordvectors")

    train_df = add_vector_col(train_df)
    dev_df = add_vector_col(dev_df)
    train_df = add_context(train_df)
    dev_df = add_context(dev_df)

    # call svm for model 1
    simple_model(train_df, dev_df)

    # call NN_model function for models 2 and 3
    NN_model(train_df, dev_df)
