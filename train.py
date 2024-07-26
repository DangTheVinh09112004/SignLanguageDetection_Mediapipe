import os
import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.preprocessing import LabelEncoder
from tqdm.autonotebook import tqdm
import pickle
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
with open("file.pickle", "rb") as f:
    data = pickle.load(f)
points = data["Points"]
labels = data["Labels"]
le = LabelEncoder()
labels = le.fit_transform(labels)
points_tensor = torch.tensor(points, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.long)


class HandGestureClassifier(nn.Module):
    def __init__(self):
        super(HandGestureClassifier, self).__init__()
        self.fc1 = nn.Linear(42, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, len(le.classes_))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class AugmentedTenSorDataset(TensorDataset):
    def __init__(self, tensors, transform=None):
        super().__init__(*tensors)
        self.transform = transform

    def __getitem__(self, item):
        x = self.tensors[0][item]
        y = self.tensors[1][item]
        if transforms:
            x = self.transform(x)
        return x, y


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HandGestureClassifier().to(device)
X_train, x_test, Y_train, y_test = train_test_split(points_tensor, labels_tensor,
                                                    test_size=0.2, random_state=42)
data_augmentation = transforms.Compose([
    transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
    transforms.Lambda(lambda x: x + 0.1),
    transforms.Lambda(lambda x: x * 1.1),
    transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1)
])
train_dataset = AugmentedTenSorDataset((X_train,Y_train), transform=data_augmentation)
test_dataset = TensorDataset(x_test, y_test)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
num_epochs = 50
best_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")
    model.eval()
    validation_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss_val = criterion(outputs, labels)
            validation_loss += loss_val.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    validation_loss /= len(test_loader)
    print("Accuracy: {:.2f}%".format(100 * correct / total))
    if validation_loss < best_loss:
        best_loss = validation_loss
        with open("model1.pickle", "wb") as f:
            pickle.dump(model.state_dict(), f)
            print("Save best model to model.pickle")