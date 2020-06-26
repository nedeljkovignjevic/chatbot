from src.model import Model

from nltk.stem.lancaster import LancasterStemmer
from nltk import word_tokenize  # function pointer

import numpy as np
import torch
import random


class Bot(object):

    def __init__(self, model: Model):
        data = np.load('data/processed.npz', allow_pickle=True)
        self.model = model
        self.vocabulary = data['arr_2']
        self.labels = data['arr_3']
        self.data = data['arr_4'][0]

    def respond(self, message: str):
        """
        Returns respond to users message
        """
        inp = self.get_bag(message)
        output = self.model(torch.from_numpy(inp).float())
        output_index = int(torch.argmax(output))

        # Need to add this when i upgrade the model
        # if output[output_index] < 0.7:
        #    return "Try again buddy. I really need some upgrades, can't understand you on this one."

        tag = self.labels[output_index]
        responses = None
        for t in self.data["intents"]:
            if t['tag'] == tag:
                responses = t['responses']
                break

        return random.choice(responses)

    def get_bag(self, text: str):
        """
        Converting users input to a bag of words
        """
        stemmer = LancasterStemmer()
        bag = [0 for _ in range(len(self.vocabulary))]

        # lower all words and take roots only
        words = word_tokenize(text)
        words = [stemmer.stem(w.lower()) for w in words]

        for word in words:
            for i, word_vocabulary in enumerate(self.vocabulary):
                if word_vocabulary == word:
                    bag[i] = 1

        return np.array(bag)
