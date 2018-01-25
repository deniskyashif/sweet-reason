import os
import numpy as np
import nltk
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from nltk.corpus import stopwords
from itertools import groupby

import xml.etree.ElementTree as ET
tree = ET.parse('./descript/cooking pasta.new.xml')
root = tree.getroot()
sentences = []

for child in root:
    for item in child:
        sentences.append(item.attrib["original"])

def load_glove(dim):
    word2vec = {}

    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                           "data/glove/glove.6B." + str(dim) + "d.txt")) as f:
        for line in f:
            l = line.split()
            word2vec[l[0]] = list(map(float, l[1:]))

    return word2vec


def sentence_vec(sentence):
    a = [np.array(word2vec[w]) for w in nltk.word_tokenize(sentence.lower()) if w not in stop_words and len(w) > 0 and w in word2vec]
    aSum = sum(a)
    return aSum / np.linalg.norm(aSum)


word2vec = load_glove(50)

stop_words = stopwords.words('english')

king = np.array(word2vec['king'])
man = np.array(word2vec['man'])
woman = np.array(word2vec['woman'])
queen = np.array(word2vec['queen'])

embedings = [sentence_vec(sentence) for sentence in sentences]

question = sentence_vec('boiled')
print(list(map(lambda x: (x[0], np.dot(x[1], question), '\n'),sorted(zip(sentences,embedings), key=lambda x: np.dot(x[1],question), reverse=True))))

Z = linkage(embedings, 'complete', 'cosine')
clusters = fcluster(Z, 0.5, criterion='distance')
print(max(clusters))
groups = []
for k,g in groupby(sorted(zip(clusters, sentences, embedings), key=lambda x: x[0]), key=lambda x: x[0]):
        groups.append(list(g))

for g in groups:
    sentences_embedings_in_group = list(map(lambda x: x[2], g ))
    sentence_sum = sum(sentences_embedings_in_group)
    sentence_avg = sentence_sum/ len(sentences_embedings_in_group)
    min_dist = 1.0
    closest_array = 0
    for i, sentence_embeding in enumerate(sentences_embedings_in_group[1:]):
        cosine_dist = np.dot(sentence_embeding,sentence_avg)
        if cosine_dist < min_dist:
            min_dist = cosine_dist
            closest_array = i

    print(g[closest_array][1])



while True:
    sentence1 = input("1:")
    sentence_vec_1 = sentence_vec(sentence1)
    sentence2 = input("2:")
    sentence_vec_2 = sentence_vec(sentence2)
    print(np.dot(sentence_vec_1, sentence_vec_2))




