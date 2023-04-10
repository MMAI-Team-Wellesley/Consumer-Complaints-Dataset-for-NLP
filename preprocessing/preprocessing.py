from collections import Counter
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from flair.data import Sentence
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings
from tqdm import tqdm
tqdm.pandas()
'''
This function cleans the text in the "narrative" column of the dataframe. 
It removes URLs, HTML tags, punctuation, numbers, stopwords, and lemmatizes the text.
'''

'''
For the data cleaning team, only edit this file. Feel free to test the functions with the dataset
in this file, but do not change the original dataset.
'''

def data_exploration(df):
    df.head(5)
    print(df.dtypes)
    # Visualize the distribution of complaints by product
    plt.figure(figsize=(10,6))
    sns.countplot(x='product', data=df)
    plt.xticks(rotation=90)
    plt.title('Distribution of Complaints by Product')
    plt.show()

    # Create a WordCloud based on the most frequent words in narrative column
    # Combine all the narratives into a single string
    narratives = ' '.join(df['narrative'].dropna())

    # Tokenize the string into words
    words = nltk.word_tokenize(narratives.lower())

    # Remove stop words from the list of words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Count the frequency of each word
    word_freq = Counter(words)

    # Create a word cloud from the most common words
    wordcloud = WordCloud(background_color='white', width=1200, height=800).generate_from_frequencies(word_freq)

    # Plot the word cloud
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

    # Identify any missing values in the dataset
    print('Missing values:\n', df.isnull().sum())

    # Print the rows with the missing values
    missing_rows = df[df.isnull().any(axis=1)]
    print(missing_rows)

def clean_text(text):
    # Remove URLs from text
    text = re.sub(r'http\S+', '', text)
    # Remove HTML tags from text
    text = re.sub(r'<.*?>', '', text)
    # Convert text to lowercase
    text = text.lower()
    # Remove contractions from text**
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    # Remove non-alphanumeric characters from text, especially the occurrence of ##**
    text = re.sub(r'[^\w\s#]', '', text).replace('#', '')
    # Remove punctuation from text
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers from text
    text = re.sub(r'\d+', '', text)
    # Remove whitespaces from text**
    text = text.strip()
    return text

def generate_tfidf_dataframe(df):
    max_features = 1000
    # Create a TfidfVectorizer object
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
    # Fit the vectorizer to the text data and transform the data to TF-IDF vectors
    tfidf_data = vectorizer.fit_transform(df['narrative'])
    feature_names = list(vectorizer.vocabulary_.keys())
    # Convert the sparse matrix to a dense matrix and create a dataframe
    tfidf_df = pd.DataFrame(tfidf_data.toarray(), columns=feature_names)
    # Concatenate the original dataframe with the new TF-IDF dataframe
    df_tfidf_clean = pd.concat([df, tfidf_df], axis=1)

    return df_tfidf_clean



def tokenize_text(text):
    return nltk.word_tokenize(text)

def load_glove_vectors(glove_file):
    embeddings = {}
    with open(glove_file, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.strip().split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

def get_glove_sentence_vector(sentence, embeddings):
    words = nltk.word_tokenize(sentence)
    words = [word for word in words if word in embeddings]
    if words:
        return np.mean([embeddings[word] for word in words], axis=0)
    else:
        return np.zeros(embeddings[next(iter(embeddings))].shape)
    
def preprocess_data(df, embeddings):
    # Drop the missing rows
    df = df.dropna()
    # Remove the "Unnamed: 0" column
    df = df.drop(columns=['Unnamed: 0'])
    # Clean the "product" column
    df['product'] = df['product'].apply(lambda x: x.lower())
    # Clean the "narrative" column
    df['narrative'] = df['narrative'].apply(lambda x: clean_text(x))
    df = generate_tfidf_dataframe(df)
    df['glove_embeddings'] = df['narrative'].progress_apply(lambda x: get_glove_sentence_vector(str(x), embeddings))
    df['tokens'] = df['narrative'].progress_apply(lambda x: tokenize_text(str(x)))
    return df

def main():
    # import the dataset "complaints_processed.csv"
    df = pd.read_csv('complaints_processed.csv')
    # pre-trained GloVe embeddings: https://nlp.stanford.edu/projects/glove/
    glove_vectors = load_glove_vectors('glove.6B.100d.txt')
    # data_exploration(df)
    df_clean = preprocess_data(df, glove_vectors)
    print(df_clean.head(5))
    # export the cleaned dataset to a csv file
    df_clean.to_csv('complaints_processed_clean_v2.csv')

if __name__ == '__main__':
    main()