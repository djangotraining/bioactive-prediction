import torch
from inf721_model import PyTorchModel
from inf721_dataset import Database


# Instantiate the PyTorch model
X_padded, y_encoded, vocab_size = Database('finalData.csv')
input_dim = vocab_size
embedding_dim = 64
hidden_dim = 32

# Creating an instance of the class
pytorch_model = PyTorchModel(input_dim, embedding_dim, hidden_dim)

# Load the saved model weights
model_weights_path = 'model_weights_epoch_10.pt'  # Adjust the filename based on your saved weights
pytorch_model.load_state_dict(torch.load(model_weights_path))
pytorch_model.eval()  # Set the model to evaluation mode

# Perform inference on new data
new_data = torch.LongTensor(X_padded[1]) 

# Add batch dimension as the first dimension
new_data = new_data.unsqueeze(0)

with torch.no_grad():
    predictions = pytorch_model(new_data)
    print(predictions)
    if predictions >= 0.5:
        print('active')
    else:
        print('Inactive')

# Convert predictions to numpy array for further analysis or use in scikit-learn
#predictions_np = predictions.numpy()
