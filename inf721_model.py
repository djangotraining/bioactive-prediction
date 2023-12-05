class PyTorchModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_units):
        super(PyTorchModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.batch_norm1 = nn.BatchNorm1d(embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, lstm_units, batch_first=True)
        self.batch_norm2 = nn.BatchNorm1d(lstm_units)
        self.dropout1 = nn.Dropout(0.5)
        self.dense1 = nn.Linear(lstm_units, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), -1)  # Flatten the sequence for Batch Normalization
        x = self.batch_norm1(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take the output of the last timestep
        x = self.batch_norm2(x)
        x = self.dropout1(x)
        x = F.relu(self.dense1(x))
        x = self.dropout2(x)
        x = torch.sigmoid(self.dense2(x))
        return x

