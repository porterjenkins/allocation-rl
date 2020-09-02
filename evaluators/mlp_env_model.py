import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
import pickle


from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import OneHotEncoder

from utils import get_store_id, get_action_space
import config.config as cfg

def get_buffer(fpath):
    with open(fpath, 'rb') as f:
        buffer = pickle.load(f)

    return buffer


def get_train_data(buffer_path, as_tensor=False, n_actions=None):

    buffer = get_buffer(buffer_path)

    s, a, r, s_prime, _ = buffer.storage[0]
    state_space = len(s)
    n_samples = len(buffer.storage)

    X = np.zeros((n_samples, state_space))
    #y = np.zeros(n_samples, n_actions)

    y = np.zeros(n_samples)

    for i in range(len(buffer.storage)):
        s, a, r, s_prime, _ = buffer.storage[i]

        X[i, :] = s
        #y[i, a] = 1.0

        if a == -1:
            a = 0

        y[i] = a





    if as_tensor:
        X = torch.from_numpy(X.astype(np.float32))
        y = torch.from_numpy(y.astype(np.int))

    return X, y




class EnvDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        sample = self.X[idx, :]
        label = self.y[idx]

        return sample, label

    def __len__(self):
        return self.X.shape[0]

class MLPClassifer(nn.Module):
    def __init__(self, n_actions, features, h_1, h_2):
        super().__init__()

        self.fc1 = nn.Linear(features, h_1)
        self.fc2 = nn.Linear(h_1, h_2)
        self.fc3 = nn.Linear(h_2, n_actions)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)

        if self.training:
            return output
        else:
            return self.softmax(output)



def main(model, dataloader, hyp, fname):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=hyp["lr"])

    for epoch in range(hyp["epochs"]):
        running_loss = 0
        for instances, labels in dataloader:
            optimizer.zero_grad()

            output = model(instances)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(running_loss / len(dataloader))


    torch.save(model, fname)




if __name__ == "__main__":
    store_id = get_store_id(cfg.vals["train_data"])
    buffer_path = f"../data/{store_id}-buffer-d-trn.p"
    action_space = get_action_space(cfg.vals["n_regions"], cfg.vals["n_products"])


    X, y = get_train_data(buffer_path, as_tensor=True, n_actions=action_space)
    train = EnvDataset(X, y)

    mlp = MLPClassifer(n_actions=action_space, features=X.shape[1], h_1=1024, h_2=512)

    hyp = {
        "epochs": 100,
        "lr": 5e-3,
        "batch_size": 32
    }

    dataset = DataLoader(train, batch_size=hyp["batch_size"])
    main(mlp, dataset, hyp, f"../data/{store_id}-env_policy.pt")