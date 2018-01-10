# Classifies a text to predict a DeScript\OMCS scenario with cosine similarity
#
# Usage:
# Import using:
# from scenario_classifier import calculate_similarity
# 
# the method expects a string
# calculate_similarity("Wash my clothes, clean the house, do the dishes, feed the dog, walk the dog and change the dog's water and throw the garbage")
#
# Output:
# list of 3 tuples ('Corpus', similarity)
#
# Exmaple output:
# [('../data/omcs_stories\\feed_a_pet_dog.xml', 0.19806299364724256), ('../data/omcs_stories\\walk_the_dog.xml', 0.15819008299418666), ('../data/omcs_stories\\remove_and_replace_garbage_bag.xml', 0.12066377914271882)]


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.stem.porter import PorterStemmer
import xml.etree.ElementTree as et
from numpy import array
import nltk
import glob
import os


def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        stems.append(PorterStemmer().stem(item))
    return stems


sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)

def import_texts(path, attribute, script_list, names_list):
    for infile in glob.glob(os.path.join(path, '*.*')):
        names_list.append(infile)
        root = et.parse(infile).getroot()
        text = ""
        for child in root:
            for i in child:
                text += i.attrib[attribute]
                text += " "
        script_list.append(text)


scripts = []
script_names = []
import_texts("../data/omcs_stories", "text", scripts, script_names)
import_texts("../data/DeScript", "original", scripts, script_names)

        
def calculate_similarity(target):
    sklearn_representation = sklearn_tfidf.fit_transform(scripts)
    sklearn_test = sklearn_tfidf.transform([target])
    cosine_similarities = linear_kernel(sklearn_test, sklearn_representation).flatten()
    related_docs_indices = cosine_similarities.argsort()[:-4:-1]
    return list(zip(array(script_names)[related_docs_indices], cosine_similarities[related_docs_indices]))