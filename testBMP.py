import os
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from trainBMP import test_loader
from trainBMP import ThaiOCRModel
from trainBMP import num_classes 
from decimal import Decimal
##### The script must allow for batching and for an arbitrary number of training epochs.
# Load the trained model
model = ThaiOCRModel(num_classes)
model.load_state_dict(torch.load('thaiocr_model.pth', weights_only=True))

# Initialize lists to store true labels and predictions
all_labels = []
all_predictions = []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
print("start test")
test_loss = 0
correct = 0
total = 0
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

# Calculate metrics
accuracy = Decimal(1 - accuracy_score(all_labels, all_predictions))
precision = Decimal(1 - precision_score(all_labels, all_predictions, average='weighted'))
recall = Decimal(1 - recall_score(all_labels, all_predictions, average='weighted'))
f1 = Decimal(1 - f1_score(all_labels, all_predictions, average='weighted'))

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')