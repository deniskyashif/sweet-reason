import xml.etree.ElementTree as ET
from instance_models import *


def get_instances(file):
    tree = ET.parse(file)
    root = tree.getroot()

    def read_question(q):
        text = q.attrib['text']
        id = q.attrib['id']
        answers = [Answer(a.attrib['id'], a.attrib['text'],a.attrib['correct'] == 'True') for a in q]

        return Question(text,answers)

    def read_instance(instance_object):
        id = instance_object.attrib['id']
        text = instance_object[0].text
        questions = []
        for question in instance_object[1]:
            questions.append(read_question(question))

        return Instance(id, text,questions)

    instances = []
    for child in root:
        instances.append(read_instance(child))

    return instances;