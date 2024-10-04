import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder

def extract_label_from_filename(filename):
    return filename.split('_')[0]  # Extract label from file name

# Generate lable
train_set_folder = '/home/guswanyie@GU.GU.SE/srv/www/train/'
train_image_paths = []
train_labels = []

for filename in os.listdir(train_set_folder):
    if filename.endswith('.bmp'):
        train_image_paths.append(os.path.join(train_set_folder, filename))
        train_labels.append(extract_label_from_filename(filename))
# print(train_labels)
test_set_folder = '/home/guswanyie@GU.GU.SE/srv/www/test/'
test_image_paths = []
test_labels = []

for filename in os.listdir(test_set_folder):
    if filename.endswith('.bmp'):
        test_image_paths.append(os.path.join(test_set_folder, filename))
        test_labels.append(extract_label_from_filename(filename))

val_set_folder = '/home/guswanyie@GU.GU.SE/srv/www/val/'
val_image_paths = []
val_labels = []

for filename in os.listdir(val_set_folder):
    if filename.endswith('.bmp'):
        val_image_paths.append(os.path.join(val_set_folder, filename))
        val_labels.append(extract_label_from_filename(filename))

class ThaiOCRDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('L')  # Convert to grayscale
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        # label = torch.tensor(label) 
        return image, label

###### The model must be trained in PyTorch 
# Simple CNN Model
class ThaiOCRModel(nn.Module):
    def __init__(self, num_classes):
        super(ThaiOCRModel, self).__init__()
        # There must be at least one hidden layer and nonlinearity in the model.
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(64 * 16 * 16, 128)  # Adjust size based on input image size
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.relu1(self.conv1(x)))
        x = self.pool(self.relu2(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)  # Flatten
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# Image transform
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # size 
    transforms.ToTensor(),
])

# Convert labels to integers
label_encoder = LabelEncoder()
#ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

train_labels = label_encoder.fit_transform(train_labels)
val_labels = label_encoder.fit_transform(val_labels)
test_labels = label_encoder.fit_transform(test_labels)
# ordinal_encoder.fit(train_labels.reshape(-1, 1))
# ordinal_encoder.transform(val_labels.reshape(-1, 1))
# ordinal_encoder.transform(val_labels.reshape(-1, 1))

# print(val_labels)
# # ordinal_encoder.fit(val_labels.reshape(-1, 1))
# # ordinal_encoder.transform(val_labels.reshape(-1, 1))

# test_labels = label_encoder.transform(test_labels)
# ordinal_encoder.fit(test_labels.reshape(-1, 1))
# ordinal_encoder.transform(test_labels.reshape(-1, 1))

# train set
train_dataset = ThaiOCRDataset(train_image_paths, train_labels, transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# val set
val_dataset = ThaiOCRDataset(val_image_paths, val_labels, transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# test set
test_dataset = ThaiOCRDataset(test_image_paths, test_labels, transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


##### The script must allow for batching and for an arbitrary number of training epochs.
# Parameters
batch_size = 32
num_epochs = 10
learning_rate = 0.001
model_save_path = 'thaiocr_model.pth'

# Initialize model, loss function, and optimizer
num_classes = len(set(train_labels))  # Number of unique labels
model = ThaiOCRModel(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

##### Training must involve exactly one GPU
# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    print(f'start training epoch{epoch}')
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # print("start validation")
    # # val
    # model.eval()
    # val_loss = 0
    # with torch.no_grad():
    #     for images, labels in val_loader:
    #         images = images.to(device)
    #         labels = labels.to(device)
    #         outputs = model(images)
    #         loss = criterion(outputs, labels)
    #         val_loss += loss.item()

    # val_loss /= len(val_loader)
    # print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}')



##### The script must save the trained model to a specified file
torch.save(model.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')
