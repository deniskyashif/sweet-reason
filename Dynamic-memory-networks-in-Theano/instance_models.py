class Instance:
    def __init__(self, id, text, questions):
        self.id = id
        self.text = text
        self.questions = questions


class Question:
    def __init__(self, text, answers):
        self.text = text
        self.answers = answers


class Answer:
    def __init__(self, id, text, isCorrect):
        self.id = id
        self.text = text
        self.isCorrect = isCorrect


class PredictedAnswer(Answer):
    def __init__(self, id, text, isCorrect, certainty):
        super().__init__(id, text, isCorrect)
        self.certainty = certainty