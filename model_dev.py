import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from generate_data import *

data_path = '/Users/emilyries/Downloads/Data_Science/Project/Implementation/data-science-24/features.parquet'

train_data, test_data, ok_eval_data = get_data(data_path)

# Convert the NumPy arrays to PyTorch tensors
train_data = torch.tensor(train_data, dtype=torch.float32)
test_data = torch.tensor(test_data, dtype=torch.float32)
ok_eval_data = torch.tensor(ok_eval_data, dtype=torch.float32)

# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
input_dim = train_data.shape[1]  # Number of features (200 in this case)
encoding_dim = 64

# Instantiate the model, define the loss function and the optimizer
model = Autoencoder(input_dim=input_dim, encoding_dim=encoding_dim)
criterion = nn.MSELoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training parameters
num_epochs = 50
batch_size = 2

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
ok_loader = torch.utils.data.DataLoader(ok_eval_data, batch_size=1, shuffle=False)

# Training loop
for epoch in range(num_epochs):
    for data in train_loader:
        # Forward pass
        output = model(data)
        loss = criterion(output, data)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Print the loss for every epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print('Training complete')

threshold = 0.1  # Adjust as needed

print('True Value is non-positive')
for i, data in enumerate(test_loader, 0):
    output = model(data)
    loss = criterion(output, data)
    if loss.item() < threshold:
        print(f'Sample {i} is likely positive.')
    if loss.item() >= threshold:
        print(f'Sample {i} is likely non-positive.')

print('True value is positive')
for i, data in enumerate(ok_loader, 0):
    output = model(data)
    loss = criterion(output, data)
    if loss.item() < threshold:
        print(f'Sample {i} is likely positive.')
    if loss.item() >= threshold:
        print(f'Sample {i} is likely non-positive.')