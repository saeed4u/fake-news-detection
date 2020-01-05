import numpy as np
import data_transformer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import LSTM
from keras.layers import Dense
import scikitplot.plotters as plot
from collections import Counter

data_transformer.clean_data()

text_train = np.load('./text_train.npy')
text_text = np.load('./text_test.npy')

label_train = np.load('./label_train.npy')
label_text = np.load('./label_text.npy')

text_train_ = []
counter = Counter()

for text in text_train:
    text_train_.append(text.split())

for word in text_train_[-1]:
    counter[word] += 1

most_common = counter.most_common(5000)
word_bank = {}
id_num = 1

for (word, _) in most_common:
    word_bank[word] = id_num
    id_num += 1

for news in text_train:
    i = 0
    while i < len(news):
        if news[i] in word_bank:
            news[i] = word_bank[news[i]]
            i += 1
        else:
            del news[i]

text_train = list(text_train)
label_train = list(label_train)

