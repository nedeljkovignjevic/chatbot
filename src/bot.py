from src.model import Model, ChatbotDataset

from nltk.stem.lancaster import LancasterStemmer
from nltk import word_tokenize  # function pointer

import numpy as np
import torch
import random


def get_bag(text: str, vocabulary: list):
    """
    Converting user input to a bag of words
    """
    stemmer = LancasterStemmer()
    bag = [0 for _ in range(len(vocabulary))]

    # lower all words and take roots only
    words = word_tokenize(text)
    words = [stemmer.stem(w.lower()) for w in words]

    for word in words:
        for i, word_vocabulary in enumerate(vocabulary):
            if word_vocabulary == word:
                bag[i] = 1

    return np.array(bag)


def main():

    # Load data
    data = np.load('data/processed.npz', allow_pickle=True)
    words = data['arr_2']
    labels = data['arr_3']
    data = data['arr_4'][0]

    # Prepare model for evaluation
    data_set = ChatbotDataset()
    model = Model(len(data_set.x[0]), len(data_set.y[0]))
    model.load_state_dict(torch.load('model/model.pth'))
    model.eval()

    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("Say something: ")
        if inp.lower() == "quit":
            break

        inp = get_bag(inp, words)
        output = model(torch.from_numpy(inp).float())
        output = int(torch.argmax(output))

        tag = labels[output]
        responses = None
        for t in data["intents"]:
            if t['tag'] == tag:
                responses = t['responses']
                break

        print(random.choice(responses))