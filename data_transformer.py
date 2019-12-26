import numpy as np
import re
import string
import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from gensim import utils
from nltk.corpus import stopwords


def text_clean(text):
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = text.lower().split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return (text)


def cleanup(text):
    text = text_clean(text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def construct_labeled_sentences(data):
    sentences = []
    for index, row in data.iteritems():
        sentences.append(LabeledSentence(utils.to_unicode(row).split(), ['Text' + '_%s' % str(index)]))
    return sentences


def get_embeddings(path, vector_dimension=300):
    data = pd.read_csv(path)

    missing_rows = []
    for i in range(len(data)):
        if data.loc[i, 'text'] != data.loc[i, 'text']:
            missing_rows.append(i)
    data = data.drop(missing_rows).reset_index().drop(['index', 'id'], axis=1)

    for i in range(len(data)):
        data.loc[i, 'text'] = cleanup(data.loc[i, 'text'])

    x = construct_labeled_sentences(data['text'])
    y = data['label'].values

    text_model = Doc2Vec(min_count=1, window=5, vector_size=vector_dimension, sample=1e-4, negative=5, workers=7,
                         epochs=10,
                         seed=1)
    text_model.build_vocab(x)
    text_model.train(x, total_examples=text_model.corpus_count, epochs=text_model.iter)

    train_size = int(0.8 * len(x))
    test_size = len(x) - train_size

    text_train_arrays = np.zeros((train_size, vector_dimension))
    text_test_arrays = np.zeros((test_size, vector_dimension))
    train_labels = np.zeros(train_size)
    test_labels = np.zeros(test_size)

    for i in range(train_size):
        text_train_arrays[i] = text_model.docvecs['Text_' + str(i)]
        train_labels[i] = y[i]

    j = 0
    for i in range(train_size, train_size + test_size):
        text_test_arrays[j] = text_model.docvecs['Text_' + str(i)]
        test_labels[j] = y[i]
        j = j + 1

    return text_train_arrays, text_test_arrays, train_labels, test_labels


def clean_data(path='datasets/train.csv', vector_dimensions=300):
    print(path)
    data = pd.read_csv(path)
    missing_rows = []
    for index in range(len(data)):
        if data.loc[index, 'text'] != data.loc[index, 'text']:
            missing_rows.append(index)
        else:
            data.loc[index, 'text'] = cleanup(data.loc[index, 'text'])

    data = data.drop(missing_rows).reset_index().drop(['index', 'id'], axis=1)
    data = data.sample(frac=1).reset_index(drop=True)

    text = data.loc[:, 'text'].values
    label = data.loc[:, 'label'].values

    train_size = int(0.8 * len(text))

    text_train = text[:train_size]
    text_test = text[train_size:]

    label_train = label[:train_size]
    label_test = label[train_size:]

    np.save('text_train.npy', text_train)
    np.save('text_test.npy', text_test)
    np.save('label_train.npy', label_train)
    np.save('label_test.npy', label_test)


clean_data()
