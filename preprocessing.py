import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

'''
This function cleans the text in the "narrative" column of the dataframe. 
It removes URLs, HTML tags, punctuation, numbers, stopwords, and lemmatizes the text.
'''

'''
For the data cleaning team, only edit this file. Feel free to test the functions with the dataset
in this file, but do not change the original dataset.
'''

def clean_text(text):
    # Remove URLs from text
    text = re.sub(r'http\S+', '', text)
    # Remove HTML tags from text
    text = re.sub(r'<.*?>', '', text)
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation from text
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers from text
    text = re.sub(r'\d+', '', text)
    # Remove stopwords from text
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    text_tokens = nltk.word_tokenize(text)
    text = [word for word in text_tokens if not word in stop_words]
    text = ' '.join(text)
    # Lemmatize text
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()
    text_tokens = nltk.word_tokenize(text)
    text = [lemmatizer.lemmatize(word) for word in text_tokens]
    text = ' '.join(text)
    return text

def preprocess_data(df):
    # Remove the "Unnamed: 0" column
    df = df.drop(columns=['Unnamed: 0'])
    # Clean the "product" column
    df['product'] = df['product'].apply(lambda x: x.lower())
    # Clean the "narrative" column
    df['narrative'] = df['narrative'].apply(lambda x: clean_text(x))
    return df

# import the dataset "complaints_processed.csv"
df = pd.read_csv('complaints_processed.csv')
test_df_clean = preprocess_data(df)