# Resolves narrator speech.
#
# Usage:
# python3 narrator_resolution.py <path-to-train-data-xml>

import sys
import xml.etree.ElementTree
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import word_tokenize
from nltk.tokenize.moses import MosesDetokenizer


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


detokenizer = MosesDetokenizer()


def resolve_narrator(text):
    tokens = word_tokenize(text)
    tags = nltk.pos_tag(tokens)

    for i in range(0, len(tokens)):
        if tags[i][1] == 'PRP' and tags[i][0] == 'I':
            tokens[i] = 'the narrator'
        elif tags[i][1] == 'PRP' and tags[i][0] == 'me':
            tokens[i] = 'the narrator'
        elif tags[i][1] == 'PRP' and tags[i][0] == 'myself':
            tokens[i] = 'the narrator'
        elif tags[i][1] == 'PRP$' and tags[i][0] == 'my':
            tokens[i] = "the narrator's"

    return detokenizer.detokenize(tokens, return_str=True)


task_train_data_path = sys.argv[1]
document_root = xml.etree.ElementTree.parse(task_train_data_path).getroot()
story = document_root.findall('instance/text')[3]

text = story.text
print(text)
print()

resolved_text = resolve_narrator(text)
print(resolved_text)
