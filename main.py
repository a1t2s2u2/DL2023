import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# データのロード
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
valid_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform)

batch_size = 256

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# Graph Layerの定義
class Graph(nn.Module):
    def __init__(self, size, d, loop):
        super(Graph, self).__init__()
        self.size = size
        self.d = d
        self.loop = loop
        shape = (size, size, 2*d+1, 2*d+1)
        low, high = -0.5, 0.5
        self.weights = nn.Parameter((high - low) * torch.rand(*shape) + low)

    def forward(self, inputs):
        _batch_size = inputs.shape[0]
        s, d = self.size, self.d
        inputs = inputs.view(_batch_size, s, s)
        updated_neurons = inputs.clone()

        for _ in range(self.loop):
            for x in range(d, s - d):
                for y in range(d, s - d):

                    v = updated_neurons[:, y, x].unsqueeze(1).unsqueeze(2)
                    updated_neurons_copy = updated_neurons.clone()
                    updated_neurons_copy[:, y, x] = 0

                    y_range = slice(y-d, y+d+1)
                    x_range = slice(x-d, x+d+1)

                    updated_neurons_copy[:, y_range, x_range] += v * self.weights[y, x].unsqueeze(0)
                    updated_neurons = updated_neurons_copy

        return updated_neurons.view(_batch_size, -1)

# モデルの定義
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28**2, 8**2),
    nn.ReLU(),
    nn.BatchNorm1d(8**2),
    Graph(size=8, d=2, loop=5),
    nn.ReLU(),
    nn.BatchNorm1d(8**2),
    nn.Linear(8**2, 10),
    nn.ReLU(),
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 学習
num_epochs = 15
total_step = len(train_loader)
result = {"valid_loss": [], "train_loss": [], "acc": []}

for epoch in range(num_epochs):
    losses = {"train": [], "valid": []}

    # 学習
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses["train"].append(loss.item())

    # 検証
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            losses["valid"].append(loss.item())

    acc = round(100 * (correct / total), 4)
    valid_loss_mean = torch.tensor(losses["valid"]).mean()
    train_loss_mean = torch.tensor(losses["train"]).mean()

    result["train_loss"].append(train_loss_mean.item())
    result["valid_loss"].append(valid_loss_mean.item())
    result["acc"].append(acc)

    print('  epoch {} train_loss: {:.4f} valid_loss: {:.4f}  acc: {:.2f}'.format(epoch + 1, train_loss_mean, valid_loss_mean, acc))