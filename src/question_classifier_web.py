'''
"High Accuracy Rule-based Question Classification using Question Syntax and Semantics":
http://www.aclweb.org/anthology/C16-1116
Web API Docs:
http://www.harishmadabushi.com/research/questionclassification/question-classification-api-documentation/
'''

import requests


def classify_question(question):
    """
    Determines the question class of a sentence.

    >>> classify_question("What is Dudley Do-Right's horse's name?")
    ('ENTY', 'animal')

    Parameters
    ----------
    question : string
        Interrogative sentence
    Returns
    -------
    Tuple (major_type, minor_type)
        Returns the question class as a tuple consisting of
        a major(general) and minor(specific) type.

    """
    payload = {"auth": "keho120l4l", "question": question}
    res = requests.get("http://qcapi.harishmadabushi.com/?", params=payload)
    res_obj = res.json()

    return (res_obj["major_type"], res_obj["minor_type"])
