import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

AP_lenght = 200

# Define custom Dataset for spike/no-spike classification
class SpikeNoSpikeDataset(Dataset):
    def __init__(self, data, labels):
        """
        Args:
            data (torch.Tensor): The input data (e.g., voltage signals).
            labels (torch.Tensor): Binary labels (1 for spike, 0 for no spike).
        """
        self.data = data
        self.labels = labels
    
    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.data)
    
    def __getitem__(self, idx):
        # Retrieve a single sample of data and its corresponding label
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label

# Simple binary classification model
class SpikeClassifier(nn.Module):
    def __init__(self):
        super(SpikeClassifier, self).__init__()
        self.fc1 = nn.Linear(data.shape[1], 64)  # Assuming input size is 100 (e.g., 100 samples of signal data)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)    # Output size is 1 for binary classification
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Sigmoid for binary classification
        return x.squeeze()

spike_vector = np.load('spikes_vector_1000.npy')
plain_vector = np.load('plain_vector_1000.npy')

# Create labels for spikes (1) and plain data (0)
spike_labels = np.ones(spike_vector.shape[0])  # Label for spikes
plain_labels = np.zeros(plain_vector.shape[0])  # Label for plain data

# Combine data and labels
data = np.concatenate((spike_vector, plain_vector), axis=0)
labels = np.concatenate((spike_labels, plain_labels), axis=0)

# Shuffle the data and create train-test split
data_train, data_test, labels_train, labels_test = train_test_split(
    data, labels, test_size=0.2, random_state=42, stratify=labels
)

# Convert to PyTorch tensors
data_train = torch.tensor(data_train, dtype=torch.float32)
labels_train = torch.tensor(labels_train, dtype=torch.float32)
data_test = torch.tensor(data_test, dtype=torch.float32)
labels_test = torch.tensor(labels_test, dtype=torch.float32)

# Create custom dataset and DataLoader for training
train_dataset = SpikeNoSpikeDataset(data_train, labels_train)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Create custom dataset and DataLoader for testing
test_dataset = SpikeNoSpikeDataset(data_test, labels_test)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)


def train_model():
    model = SpikeClassifier()
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 50
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_dataloader)}')

    print("Training complete.")

    # Save the trained model
    torch.save(model.state_dict(), 'model.pth')  # Save the model's state dict
    print("Model saved to model.pth")

if __name__ == "__main__":
    train_model() 
