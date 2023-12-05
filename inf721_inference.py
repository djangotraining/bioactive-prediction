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

# Create an instance of the model
pytorch_model = PyTorchModel(vocab_size, embedding_dim, lstm_units, output_dim)

# Load the saved model weights
model_weights_path = 'model_weights_epoch_30.pt'  # Adjust the filename based on your saved weights
pytorch_model.load_state_dict(torch.load(model_weights_path))
pytorch_model.eval()  # Set the model to evaluation mode

# Perform inference on new data
new_data = torch.LongTensor(new_sequences)  # Assuming new_sequences is your new data
with torch.no_grad():
    predictions = pytorch_model(new_data)

