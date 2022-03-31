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

    def start(self, data, data_test):
        n_epoch = 10

        X = []
        y = []
        for item in data:
            X.append(item[:-1])
            y.append(item[-1])
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)

        for i, epoch in enumerate(range(n_epoch)):
            y_predicted = self.model(X)

            loss = self.criterion(y_predicted, y)
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            print(f'epoch: {i}, loss: {loss.item():.4f}')
        self.model.save()

        X = []
        for item in data_test:
            X.append(item)
        X = torch.FloatTensor(X)
        y_predicted = self.model(X)
        print(y_predicted)
        print(1)









