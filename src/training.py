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
