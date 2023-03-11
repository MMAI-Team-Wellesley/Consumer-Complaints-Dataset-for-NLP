from collections import Counter
import pandas as pd
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

def preprocess_data(df):
    # Drop the missing rows
    df = df.dropna()
    # Remove the "Unnamed: 0" column
    df = df.drop(columns=['Unnamed: 0'])
    # Clean the "product" column
    df['product'] = df['product'].apply(lambda x: x.lower())
    # Clean the "narrative" column
    df['narrative'] = df['narrative'].apply(lambda x: clean_text(x))
    df = generate_tfidf_dataframe(df)
    return df

def main():
    # import the dataset "complaints_processed.csv"
    df = pd.read_csv('complaints_processed.csv')
    data_exploration(df)
    df_clean = preprocess_data(df)
    print(df_clean.head(5))
    # export the cleaned dataset to a csv file
    df_clean.to_csv('complaints_processed_clean.csv')

if __name__ == '__main__':
    main()