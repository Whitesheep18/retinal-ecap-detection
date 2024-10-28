import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split
from train import test_dataloader, SpikeClassifier  # Adjust import based on your file structure


# Load the trained model
model = SpikeClassifier()
model.load_state_dict(torch.load('model.pth'))  # Load your trained model's state dict
model.eval()  # Set the model to evaluation mode

# Make predictions and calculate accuracy on the test set
correct_predictions = 0
total_predictions = 0

with torch.no_grad():  # Disable gradient calculation
    for inputs, labels in test_dataloader:
        outputs = model(inputs)  # Get model outputs
        predictions = (outputs > 0.5).float()  # Convert outputs to binary predictions (0 or 1)

        # Ensure labels are of the same shape
        correct_predictions += (predictions.view(-1) == labels).sum().item()  # Count correct predictions
        total_predictions += labels.size(0)  # Total predictions made

# Calculate accuracy
if total_predictions > 0:
    accuracy = correct_predictions / total_predictions


print(f'Test Accuracy: {accuracy * 100:.2f}%')
