import torch
import torch.nn as nn


class PyTorchModel(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim):
        super(PyTorchModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)


    def forward(self, x):        
        embeds = self.embedding(x)   
        lstm_out, _ = self.lstm(embeds)        
        last_hidden_state = lstm_out[:,-1]
        tag_scores = torch.sigmoid(self.fc(last_hidden_state))

        return tag_scores
        

