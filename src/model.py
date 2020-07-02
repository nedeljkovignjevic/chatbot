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
        super(Model, self).__init__()
        data = np.load('data/processed.npz')
        self.inp = data['arr_0']
        self.out = data['arr_1']

        self.fc1 = keras.Input(shape=(len(self.inp[0]),))
        self.fc2 = keras.layers.Dense(4, activation="linear")
        self.fc3 = keras.layers.Dense(4, activation="linear")
        self.fc4 = keras.layers.Dense(len(self.out[0]), activation='softmax')

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.fc4(x)

    def train_model(self):
        """
        Build, train and save the model
        """
        self.compile(loss='categorical_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy'])

        self.fit(self.inp, self.out,
                 batch_size=8, epochs=500, verbose=1)

        self.save('model/model_new.h5')
