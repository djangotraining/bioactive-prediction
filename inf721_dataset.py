import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# PyTorch Dataset
class SMILESDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.LongTensor(sequences)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]



class Database:
    # Class variables (shared among all instances)
    class_variable = "I am a class variable"

    # Constructor method (called when an instance is created)
    def __init__(self, dataset):
        # Instance variables (unique to each instance)
        self.dataset = dataset
        df = pd.read_csv(self.dataset)
        self.X = df['canonical_smiles']
        self.y = df['class']

    # Instance method
    def __getitem__(self, idx):
        return self.X, self.y





