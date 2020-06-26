"""
MIT License

chatbot is relatively simple AI-based software application that simulates human conversation through text chats.
This file is part of chatbot.

Copyright (c) 2020 Nedeljko Vignjević

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
"""

import numpy as np
import json

from nltk.stem.lancaster import LancasterStemmer
from nltk import word_tokenize  # function pointer


def process_data():
    """
    Processing and saving data for neural network.
    """
    stemmer = LancasterStemmer()

    words = []  # all known words (vocabulary)
    labels = []  # all known tags
    data_x = []  # tokenized patterns (words)
    data_y = []  # tags for tokenized patterns

    data = None
    with open("data/data.json") as file:
        data = json.load(file)

        for intent in data["intents"]:
            for pattern in intent["patterns"]:
                temp_words = word_tokenize(pattern)
                words.extend(temp_words)
                data_x.append([stemmer.stem(w.lower()) for w in temp_words])
                data_y.append(intent["tag"])

            if intent["tag"] not in labels:
                labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]  # take roots only
    words = sorted(list(set(words)))
    labels = sorted(labels)

    input = []
    output = []

    # do one hot encoding
    out_empty = [0 for _ in range(len(labels))]

    for i, dat in enumerate(data_x):
        bag = []

        for w in words:
            bag.append(1) if w in dat else bag.append(0)

        out_row = out_empty[:]
        out_row[labels.index(data_y[i])] = 1

        input.append(bag)
        output.append(out_row)

    data = [data]
    input, output, words, labels, data = np.array(input), np.array(output), np.array(words), np.array(labels), np.array(data)

    np.savez("data/processed.npz", input, output, words, labels, data)
