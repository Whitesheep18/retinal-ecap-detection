import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split
from train import test_dataloader, SpikeNoSpikeDataset, SpikeClassifier 
import matplotlib.pyplot as plt

# Load your model
model = SpikeClassifier()
model.load_state_dict(torch.load('model.pth'))
model.eval()  # Set the model to evaluation mode

# Load unlabeled data
unlabeled_data = np.load('pred_data_vector_1000.npy')  # Replace with your actual data
unlabeled_data = torch.tensor(unlabeled_data, dtype=torch.float32)

# Prepare a DataLoader for the unlabeled data
unlabeled_dataset = SpikeNoSpikeDataset(unlabeled_data, labels=None)  # No labels
unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=32, shuffle=False)

# Making predictions
predictions = []
with torch.no_grad():
    for inputs in unlabeled_dataloader:
        outputs = model(inputs)
        predictions.append(outputs)

# Concatenate predictions to a single tensor
predictions = torch.cat(predictions).numpy()  # Shape: (num_samples, 1)

# Convert predictions to binary
binary_predictions = (predictions > 0.5).astype(int)  # 0 or 1

# Separate data into predicted classes
predicted_class_0 = unlabeled_data[binary_predictions.flatten() == 0]
predicted_class_1 = unlabeled_data[binary_predictions.flatten() == 1]


# Randomly select 10 samples from each class
indices_class_0 = np.random.choice(len(predicted_class_0), 10, replace=False)
indices_class_1 = np.random.choice(len(predicted_class_1), 10, replace=False)

# Get the actual samples
samples_class_0 = predicted_class_0[indices_class_0]
samples_class_1 = predicted_class_1[indices_class_1]

# Set up the figure for plotting
plt.figure(figsize=(15, 8))

# Plot samples classified as 0 (no spike)
for i, sample in enumerate(samples_class_0):
    plt.subplot(4, 5, i + 1)  # 4 rows, 5 columns for 20 total subplots
    plt.plot(sample.numpy(), label='Predicted: 0', color='blue')
    plt.axhline(y=0, color='r', linestyle='--')  # Horizontal line for reference
    plt.title('Pred: 0')
    plt.legend()

# Plot samples classified as 1 (spike)
for i, sample in enumerate(samples_class_1):
    plt.subplot(4, 5, i + 11)  # 4 rows, 5 columns (starting from 11)
    plt.plot(sample.numpy(), label='Predicted: 1', color='orange')
    plt.axhline(y=0, color='r', linestyle='--')  # Horizontal line for reference
    plt.title('Pred: 1')
    plt.legend()

plt.tight_layout()
plt.show()
