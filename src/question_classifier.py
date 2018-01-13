from enum import Enum
from pycorenlp import StanfordCoreNLP
from nltk.tree import Tree


class QuestionClass(Enum):
    NONE = 0
    ABBR = 1
    DESC = 2
    ENTY = 3
    HUM = 4
    LOC = 5
    NUM = 6


# Reference: https://github.com/kimduho/nlp/wiki/Part-of-Speech-tags
WH_PHRASE_LABELS = ["WHADJP", "WHAVP", "WHNP", "WHPP"]
WH_WORD_LABELS = ["WDT", "WP", "WP$", "WRB"]

# Running the Stanford CoreNLP Server
# https://stanfordnlp.github.io/CoreNLP/corenlp-server.html#getting-started
NLP = StanfordCoreNLP('http://localhost:9000')


def _find_in_parse_tree(node, predicate):
    if not isinstance(node, Tree):
        return False

    if predicate(node):
        return node

    for child in node:
        match = _find_in_parse_tree(child, predicate)
        if match:
            return match

    return False


def classify_question(question):
    q_class = QuestionClass.NONE
    output = NLP.annotate(question, properties={
        "annotators": "parse",
        "outputFormat": "json"
    })

    tree = Tree.fromstring(output['sentences'][0]['parse'])
    tree.pretty_print()

    wh_word_node = _find_in_parse_tree(
        tree, lambda x: x.label() in WH_WORD_LABELS)

    if wh_word_node:
        wh_word = wh_word_node.leaves()[0]
        print(wh_word)
    else: print('no')
    # TODO: Apply the classification rules

    return q_class


classify_question("What is Dudley Do-Right's horse's name?")
