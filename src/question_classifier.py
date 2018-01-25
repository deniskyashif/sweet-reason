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
WH_PHRASE_LABELS = ["WHADJP", "WHADVP", "WHNP", "WHPP"]
WH_WORD_LABELS = ["WDT", "WP", "WP$", "WRB"]

WH_WORD_TO_TYPE_MAP = {
    "what": QuestionClass.ENTY,
    "when": QuestionClass.DESC,
    "where": QuestionClass.LOC,
    "which": QuestionClass.ENTY,
    "who": QuestionClass.HUM,
    "whom": QuestionClass.HUM,
    "whose": QuestionClass.HUM,
    "why": QuestionClass.DESC,
    "how": QuestionClass.DESC
}

# Running the Stanford CoreNLP Server
# https://stanfordnlp.github.io/CoreNLP/corenlp-server.html#getting-started
NLP = StanfordCoreNLP('http://localhost:9000')


def find_in_parse_tree(node, predicate):
    if not isinstance(node, Tree):
        return False

    if predicate(node):
        return node

    for child in node:
        match = find_in_parse_tree(child, predicate)
        if match:
            return match

    return False


def find_by_label(node, label):
    return find_in_parse_tree(node, lambda x: x.label == label)


def classify_question(question):
    output = NLP.annotate(question, properties={
        "annotators": "parse",
        "outputFormat": "json"
    })

    tree = Tree.fromstring(output['sentences'][0]['parse'])
    tree.pretty_print()

    # TODO: Perform Head Noun Analysis

    # Wh* Word Analysis
    wh_word_node = find_in_parse_tree(
        tree, lambda x: x.label() in WH_WORD_LABELS)

    if wh_word_node:
        wh_word = (wh_word_node.leaves()[0]).lower()
        # TODO: Cover more specific cases
        return WH_WORD_TO_TYPE_MAP[wh_word]

    # TODO: Perform Main Verb Analysis
    return QuestionClass.NONE


# print(classify_question("What is Dudley Do-Right's horse's name?"))
