"""
Example of how to create custom dataset in Pytorch. 

In this demo we have images of cats and dogs in a folder and a csv
file containing the name to the jpg file as well as the target
label (0 for cat, 1 for dog).
"""
import os

import pandas as pd
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

from skimage import io
from torch.utils.data import DataLoader, Dataset

os.chdir(f"{os.getcwd()}/custom_dataset_demo")


class CatsAndDogDataSet(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)  # csv file in the same folder
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        # print(f"----->{img_path}")
        image = io.imread(img_path)

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(int(self.annotations.iloc[index, 1]))

        return image, label


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load our data
train_set = CatsAndDogDataSet(
    csv_file="cats_dogs.csv",
    root_dir="cats_dogs_resized",
    transform=transforms.ToTensor(),
)

# hyperparameters
in_chanel = 3
num_classes = 2
learning_rate = 1e-3
batch_size = 32
num_epochs = 10

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

# pretrained Model
model = torchvision.models.googlenet(pretrained=True)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    total_loss = 0

    for batch in train_loader:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        # forward
        preds = model(images)
        loss = F.cross_entropy(preds, labels)

        total_loss += loss.item()
        # backward
        optimizer.zero_grad()
        loss.backward()
        # update parameters
        optimizer.step()

    print(f"epoch: {epoch+1},  loss: {total_loss}")

    # check accuracy on training set to see the performance of our model


def check_accuracy(loader, model):
    model.eval()
    num_correct = 0
    num_images = 0

    with torch.no_grad():
        for batch in loader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            preds = model(images)
            num_images += len(labels)
            num_correct += preds.argmax(dim=1).eq(labels).sum().item()
        print(f"{num_correct} / {num_images}, accuracy {(num_correct/num_images):.4f}")

    model.train()


print("checking accuracy on Training set")
check_accuracy(train_loader, model)
