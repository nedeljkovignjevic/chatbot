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
from tensorflow import keras


class Model(keras.Model):

    def __init__(self):
        data = np.load('data/processed.npz')
        self.input = data['arr_0']
        self.output = data['arr_1']

        self.fc1 = keras.Input(shape=(len(self.input[0]),))
        self.fc2 = keras.layers.Dense(4, activation="linear")
        self.fc3 = keras.layers.Dense(4, activation="linear")
        self.fc4 = keras.layers.Dense(len(self.output[0]), activation='softmax')

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

    def train_model(self):
        """
        Build and save the model
        """
        self.compile(loss='categorical_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy'])

        self.fit(self.input, self.output,
                 batch_size=8, epochs=500, verbose=1)

        self.save('model/model.h5')
