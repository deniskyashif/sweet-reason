# Finds the matching DeScript scenario for a training data text.
#
# Usage:
# python3 training_data_similarity.py <path-to-train-data-xml> <path-to-descript-esds-dir>
#
# Output:
# <training-text-index> <scenario-score> <scenario-name>

import sys
import xml.etree.ElementTree
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import *

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

stemmer = PorterStemmer()

task_train_data_path = sys.argv[1]
descript_dir_path = sys.argv[2]

document_root = xml.etree.ElementTree.parse(task_train_data_path).getroot()

def is_noun(tag):
    return tag[1].startswith('NN')


def is_verb(tag):
    return tag[1].startswith('VB')


def add_items(description, text):
    tokens = word_tokenize(text)
    tags = nltk.pos_tag(tokens)

    for tag in tags:
        if is_noun(tag) or is_verb(tag):
            description.add(tag[0].lower())


story_descriptions = []
for story in document_root.findall('instance/text'):
    description = set()
    add_items(description, story.text)
    story_descriptions.append(description)


scenario_descriptions = {}
for scenario_file in os.listdir(descript_dir_path):
    scenario_path = os.path.join(descript_dir_path, scenario_file)
    scenario_root = xml.etree.ElementTree.parse(scenario_path).getroot()

    description = set()
    for item in scenario_root.findall('script/item'):
        line = item.attrib['original']
        add_items(description, line)

    scenario_name, _ = os.path.splitext(scenario_file)
    scenario_descriptions[scenario_name] = description


for idx, story_description in enumerate(story_descriptions):
    score = 0
    scenario = None

    for scenario_name, scenario_description in scenario_descriptions.items():
        common_items = len(story_description & scenario_description)
        all_items = len(story_description | scenario_description)
        scenario_score = common_items / all_items
        if scenario_score > score:
            scenario = scenario_name
            score = scenario_score

    print(idx, score, scenario)
