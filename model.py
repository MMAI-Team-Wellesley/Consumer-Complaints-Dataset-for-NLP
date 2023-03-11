import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow import keras

'''
This file should only be modified by the model building team in case of merge conflicts.
'''

# Read the data into a dataframe
# run preprocessing.py first to generate the csv file
df = pd.read_csv('complaints_processed_clean.csv')
df.head(5)
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
model = build_model(df)