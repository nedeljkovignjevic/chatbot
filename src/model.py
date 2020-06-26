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
