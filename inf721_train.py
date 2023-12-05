import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

from inf721_dataset import Database, SMILESDataset
from inf721_model import PyTorchModel

dataset = 'finalData.csv'

# Creating an instance of the class
X, y = Database(dataset)


# Tokenize the SMILES strings
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(X)
X_encoded = tokenizer.texts_to_sequences(X)

# Pad sequences to have the same length
X_padded = pad_sequences(X_encoded)

# Label encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X_padded, y_encoded, test_size=0.2, random_state=42)

# PyTorch Datasets and Dataloaders
train_dataset = SMILESDataset(X_train, y_train)
test_dataset = SMILESDataset(X_test, y_test)



batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Assuming tokenizer is a variable that holds your tokenizer
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 64
lstm_units = 32

# Instantiate the PyTorch model
pytorch_model = PyTorchModel(vocab_size, embedding_dim, lstm_units)

# PyTorch Loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(pytorch_model.parameters(), lr=0.001)

# Lists to store training and validation loss values
train_losses = []
val_losses = []

# Training loop
epochs = 30
for epoch in range(epochs):
    pytorch_model.train()
    for inputs, labels in train_loader:
        inputs, labels = Variable(inputs), Variable(labels)      
        pytorch_model.zero_grad()
        optimizer.zero_grad()
        outputs = pytorch_model(inputs)
        labels = labels.unsqueeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Calculate training loss
    pytorch_model.eval()
    with torch.no_grad():
        train_outputs = pytorch_model(torch.LongTensor(X_train))
        train_loss = log_loss(y_train, train_outputs.numpy())
        train_losses.append(train_loss)

        # Calculate validation loss
        test_outputs = pytorch_model(torch.LongTensor(X_test))
        val_loss = log_loss(y_test, test_outputs.numpy())
        val_losses.append(val_loss)

    print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss}, Val Loss: {val_loss}')
    
    # Save the model weights
    model_weights_path = f'model_weights_epoch_{epoch + 1}.pt'
    torch.save(pytorch_model.state_dict(), model_weights_path)
    print(f'Model weights saved to {model_weights_path}')


# Optionally, save the entire model
torch.save(pytorch_model, 'full_model.pth')
print('Full model saved')
