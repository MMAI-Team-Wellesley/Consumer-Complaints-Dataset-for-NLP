import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from preprocessing import clean_text, preprocess_data
import tensorflow as tf
from tensorflow import keras

# Use the clean_text function to clean the input data
text = "This is some sample text."
cleaned_text = clean_text(text)

# Use the preprocess_data function to preprocess the input data
df = pd.read_csv('data.csv')
preprocessed_df = preprocess_data(df)

'''
Sample code for building the model. Feel free to change the model architecture.
'''
def build_model(df):
    # Create the model
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(df.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    # Compile the model
    model.compile(loss='mean_squared_error',
                optimizer=tf.keras.optimizers.RMSprop(0.001),
                metrics=['mean_absolute_error', 'mean_squared_error'])
    return model


# Use the preprocessed data to build the model
model = build_model(preprocessed_df)