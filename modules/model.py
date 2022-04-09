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
        self.act1 = nn.Sigmoid()
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        # x = self.act1(self.linear1(x))
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
        n_epoch = 40

        for i, epoch in enumerate(range(n_epoch)):
            X, Y = self.shuffle_train()
            for j, (x, y) in enumerate(zip(X, Y)):
                y_predicted = self.model(x)

                loss = self.criterion(y_predicted, y)
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                print(f'epoch: {i}-{j} batch, loss: {loss.item():.4f}')
        self.model.save()

        X, y = [], []
        for item in self.test:
            X.append(item.features)
            y.append(item.score)
        X = torch.FloatTensor(X)
        y_predicted = self.model(X)
        answers = y_predicted.detach().numpy()

        for i, item in enumerate(self.test):
            print(f'{item.name}: prediction = {answers[i]}, our_score = {y[i]}')

        mse = self.mse(answers, y)
        print(f'MSE: {mse}')

    def shuffle_train(self):
        random.shuffle(self.train)

        X, Y = [], []
        x, y = [], []
        batch_size = 8
        for i, item in enumerate(self.train):
            x.append(item.features)
            y.append([item.score])
            if i % batch_size == batch_size - 1:
                X.append(torch.FloatTensor(x))
                Y.append(torch.FloatTensor(y))
                x.clear()
                y.clear()
        if len(x) > 0:
            X.append(torch.FloatTensor(x))
        if len(y) > 0:
            Y.append(torch.FloatTensor(y))
        return X, Y

    @staticmethod
    def mse(answers, y):
        summation = 0
        for i in range(len(answers[0])):
            difference = answers[0][i] - y[i]
            squared_difference = difference ** 2
            summation = summation + squared_difference
        mse = summation / len(answers)
        return mse







