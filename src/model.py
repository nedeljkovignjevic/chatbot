import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class Model(nn.Module):

    def __init__(self, input_dims, output_dims):
        """
        Fully connected neural network model
        """
        super(Model, self).__init__()

        self.fc1 = nn.Linear(input_dims, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, output_dims)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.softmax(x, dim=-1)

    def evaluate(self):
        self.load_state_dict(torch.load('model/model.pth'))
        self.eval()


class ChatbotDataset(Dataset):
    """
    Chat-bot data-set model
    """
    def __init__(self):
        data = np.load('data/processed.npz')
        self.x = data['arr_0']
        self.y = data['arr_1']

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]
