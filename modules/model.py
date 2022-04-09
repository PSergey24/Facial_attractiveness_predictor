import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class LinearModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class Trainer:

    def __init__(self, model):
        self.lr = 0.001
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        self.train = None
        self.test = None

    def start(self, train, test):
        self.train, self.test = train, test
        n_epoch = 80

        for i, epoch in enumerate(range(n_epoch)):
            X, y = self.shuffle_train()
            y_predicted = self.model(X)

            loss = self.criterion(y_predicted, y)
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            print(f'epoch: {i}, loss: {loss.item():.4f}')
        self.model.save()

        X, y = [], []
        for item in self.test:
            X.append(item.features)
            y.append(item.score)
        X = torch.FloatTensor(X)
        y_predicted = self.model(X)

        for i, item in enumerate(self.test):
            print(f'{item.name}: prediction = {y_predicted[i]}, score = {y[i]}')

    def shuffle_train(self):
        random.shuffle(self.train)
        X, y = [], []
        for item in self.train:
            X.append(item.features)
            y.append([item.score])
        return torch.FloatTensor(X), torch.FloatTensor(y)








