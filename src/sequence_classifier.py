# Classifies a text to predict a DeScript\OMCS sequence with cosine similarity
#
# Usage:
# Import using: from sequence_classifier import get_sequence
# Before importing, make sure you have w2v pre-trained model inside line 32:
# w2v = gensim.models.KeyedVectors.load_word2vec_format("../data/wiki.en.vec", binary=False)
# from scenario_classifier import calculate_similarity
# 
# the method expects a string, preferably the question concatenated with the answers
# get_sequence("Did the tub fill with water? yes no")
#
# Output:
# a string that is the closest sequence from DeScript/OMCS
#
# Exmaple:
# get_sequence("Did the tub fill with water? yes no")
# "Walk into my bathroom Turn water in tub on Feel the water to see if it's the right temperature Put clog in bottom of bathtub Let tub fill up Undress Get in tub Wash myself Empty bath Towel off "


import re
import os
import glob
import gensim
import numpy as np
import xml.etree.ElementTree as et
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer

print("==> Loading wiki vectors")
w2v = gensim.models.KeyedVectors.load_word2vec_format("../data/wiki.en.vec", binary=False)
print("==> Loading wiki vectors end")

def tokenize(text):
  tokens = word_tokenize(text)
  stems = []
  for item in tokens:
    stems.append(PorterStemmer().stem(item))
  return stems


def get_sent_vector(text):
	# Expects a string that will be converted into a w2v array with additional IDF multiplication of the words
  # Returns a 300-dimentional word2vec array
  text = re.sub('(\?|\!|\.)', ' ', text)
  tokenized = text if text == "" else word_tokenize(text)
  
  all_word_vectors = []
  for w in tokenized:
    if (w in w2v):
      if (PorterStemmer().stem(w) in sklearn_tfidf.vocabulary_):
        temp_array = np.array(w2v[w])*sklearn_tfidf.idf_[sklearn_tfidf.vocabulary_[PorterStemmer().stem(w)]]
      else:
        temp_array = np.array(w2v[w])
      all_word_vectors.append(temp_array)
  if len(all_word_vectors) == 0:
    result = np.zeros(300)
  else:
    result = np.mean(all_word_vectors, axis=0)
  return [float(i) for i in result]


def get_text_vector(text):
	# splits the text on sentences and creates a mean vector
  sentences = text.split('.')
  if (sentences[-1] == ""):
    sentences = sentences[:-1]
  return np.mean([get_sent_vector(sentence) for sentence in sentences], axis=0) 


def vectorize(texts):
  return [get_text_vector(text) for text in texts]


def get_sim(a, b):
  #Computes the cosine similarity between the w2v vectors of two documents
  #Returns float in [0,1]
  sim = 1 - cosine(a, b)
  if np.isnan(sim):
    sim = 0
  return sim


def import_sequences(path, attribute):
	# Imports the common sense corpuses.
	# Expects a path and an attribute for the second level of nesting
	# Returns an array of sequences
  sequences = []
  for infile in glob.glob(os.path.join(path, '*.*')):
    root = et.parse(infile).getroot()
    text = ""
    for child in root:
      for i in child:
        text += i.attrib[attribute]
        if text[-1] != '.':
          text += "."
        text += " "
      sequences.append(text)
      text = ""
  return sequences


def import_scripts(path, attribute):
  scripts_list, names_list = [], []
  for infile in glob.glob(os.path.join(path, '*.*')):
    names_list.append(infile)
    root = et.parse(infile).getroot()
    text = ""
    for child in root:
      for i in child:
        text += i.attrib[attribute]
        text += " "
    scripts_list.append(text)
  return scripts_list, names_list


def calculate_closest(array, target):
	# Calculates the similarity between a list of sequences and a vectorized target
  similarities = []
  for vector in array:
    similarities.append(get_sim(vector, target))
  closest_index = (np.array(similarities).argsort()[-1]).tolist()
  # for ind in closest_indices:
    # print(similarities[ind])
    # print(sequences[ind])
  return closest_index


def get_sequence(question):
	return sequences[calculate_closest(vectorized_sequences, get_sent_vector(question))]


sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)

omcs_sequences = import_sequences("../data/omcs_stories", "text")
descript_sequences = import_sequences("../data/DeScript", "original")
rkp_sequences = import_sequences("../data/rkp_xml", "text")

omcs_scripts, omcs_script_names = import_scripts("../data/omcs_stories", "text")
descript_scripts, descript_script_names = import_scripts("../data/DeScript", "original")
rkp_scripts, rkp_script_names = import_scripts("../data/rkp_xml", "text")

scripts = omcs_scripts + descript_scripts + rkp_scripts
script_names = omcs_script_names + descript_script_names + rkp_script_names

sequences = omcs_sequences + descript_sequences + rkp_sequences

sklearn_representation = sklearn_tfidf.fit_transform(scripts)

vectorized_sequences = vectorize(sequences)

