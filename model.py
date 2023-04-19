from torch import nn, cuda, optim, tensor
from data_loader import get_mnist_loaders
import numpy as np
import matplotlib.pyplot as plt
import torch

device = 'cuda' if cuda.is_available() else 'cpu'
(train_loader, test_loader) = get_mnist_loaders(100)

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=28*28, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=10),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)

        return logits

model = Model().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

def train_model():
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = tensor(labels).to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            loss = loss.item()
            current = (i + 1) * len(inputs)

            print(f'loss: {loss:.4f} [{current:>5d}/{len(train_loader):>5d}]')

def eval_model():
    val_loss: list[float] = []
    val_acc: list[float] = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = tensor(labels).to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            val_loss.append(loss.item())
            val_acc.append((outputs.argmax(1) == labels).type(torch.float).mean().item())

            print(f'val_loss: {np.mean(val_loss):.4f}, val_acc: {np.mean(val_acc):.4f}')

    plt.plot(val_loss, label='val_loss')

if __name__ == '__main__':
    epochs = 5

    for epoch in range(1, epochs + 1):
        print(f'epoch {epoch}/{epochs}')

        model.train()
        train_model()

    for epoch in range(1, epochs + 1):
        print(f'epoch {epoch}/{epochs}')

        model.eval()
        eval_model()
