"""
Copyright (C) 2020 Nedeljko Vignjević

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
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
