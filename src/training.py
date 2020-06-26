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

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.model import Model, ChatbotDataset


def train():
    dataset = ChatbotDataset()
    training_loader = DataLoader(dataset, batch_size=8, shuffle=False)
    model = Model(len(dataset.x[0]), len(dataset.y[0]))

    optimizer = optim.Adam(model.parameters())
    loss_function = torch.nn.MSELoss()
    n_epochs = 1000

    model.train()

    for epoch in range(n_epochs):
        full_loss = 0
        n_loss = 0
        for input, target in training_loader:
            input, target = input.float(), target.float()

            optimizer.zero_grad()
            output = model(input)

            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()

            full_loss += loss.item()
            n_loss += 1

        print(f'{epoch}: {full_loss / n_loss}')

    torch.save(model.state_dict(), 'model/model.pth')
