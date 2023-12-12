from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pandas as pd



def Database(dataset):

    df = pd.read_csv(dataset)
    X = df['canonical_smiles']
    y = df['class']


    # Tokenize the SMILES strings
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(X)
    X_encoded = tokenizer.texts_to_sequences(X)

    vocab_size = len(tokenizer.word_index) + 1

    # Pad sequences to have the same length
    X_padded = pad_sequences(X_encoded)

    # Label encode the target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    
   
    return X_padded, y_encoded, vocab_size










