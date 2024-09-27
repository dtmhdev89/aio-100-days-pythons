import pandas as pd
import numpy as np
import sys

from bs4 import BeautifulSoup
import contractions
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

import seaborn as sns
import matplotlib.pyplot as plt

from dataset_man import dataset_manager

stop = set(stopwords.words('english'))

def expand_contractions(text):
    return contractions.fix(text)

def preprocess_text(text):
    wl = WordNetLemmatizer()
    soup = BeautifulSoup(text, "html.parser")
    # print('----text before soup html parser:\n', text)
    text = soup.get_text()
    # print('----text after soup html parser:\n', text)
    text = expand_contractions(text)
    # print('----text after contractions:\n', text)
    emoji_cleaner = re.compile("["
                            u"\U0001F600-\U0001F64F" # emoticons unicode chars)
                            u"\U0001F300-\U0001F5FF" # symbols and pictographs
                            u"\U0001F680-\U0001F6FF" # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF" # flags (iOS)
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            "]+", flags=re.UNICODE)
    text = emoji_cleaner.sub(r'', text)
    # print('----text after sub emoji and symbols to empty space:\n', text)
    text = re.sub(r'\.(?=\S)', '. ', text)
    # print('----text after sub (. without space) with (. with space):\n', text)
    text = re.sub(r'http\S+', '', text)
    # print('----text after remove url:\n', text)
    text = "".join([
        word.lower() for word in text if word not in string.punctuation
    ])
    # print('----text after lower:\n', text)
    text = " ".join([
        wl.lemmatize(word) for word in text.split() if word not in stop and word.isalpha()
    ])
    # print('----text after lemmatize:\n', text)

    return text

# Create autocpt arguments
def func(pct, allvalues):
    abosolute = int(pct / float(100) * np.sum(allvalues))
    return "{:.1f}%\n({:d})".format(pct, abosolute)

def main():
    imdb_data = dataset_manager.load_gdrive_dataset('1nxR07ebVNc5bSgfTQjeUcAoyoaNuuH6s')

    df = pd.DataFrame(imdb_data)
    print(df.head())
    print(df.shape)
    print(df[df.duplicated(keep=False)])
    print(df.duplicated().sum())

    df = df.drop_duplicates()

    df['review'] = df['review'].apply(preprocess_text)
    print(df['review'].shape)

    freq_pos = len(df[df['sentiment'] == 'positive'])
    freq_neg = len(df[df['sentiment'] == 'negative'])

    data = [freq_pos, freq_neg]

    labels = ['positive', 'negative']

    # create pie chart
    pie, ax = plt.subplots(figsize=[11, 7])
    plt.pie(x=data,
            autopct=lambda pct: func(pct, data),
            explode=[0.0025] * 2,
            pctdistance=0.5,
            colors=[sns.color_palette()[0], 'tab:red'],
            textprops={'fontsize': 16})

    labels = [r'Positive', r'Negative']
    plt.legend(labels, loc='best', prop={'size': 14})
    pie.savefig("PieChart.png")
    plt.show()

    # Analysis
    words_len = df['review'].str.split().map(lambda x: len(x))
    df_temp = df.copy()
    df_temp['words length'] = words_len

    hist_positive = sns.displot(
        data=df_temp[df_temp['sentiment'] == 'positive'],
        x="words length", hue='sentiment', kde=True, height=7, aspect=1.1, legend=False
    ).set(title="Words in positive reviews")
    plt.show()

    hist_negative = sns.displot(
        data=df_temp[df_temp['sentiment'] == 'negative'],
        x="words length", hue='sentiment', kde=True, height=7, aspect=1.1, legend=False, palette=['red']
    ).set(title="Words in negative reviews")
    plt.show()

    plt.figure(figsize=(7, 7.1))
    kernel_distribution_number_words_plot = sns.kdeplot(
        data=df_temp, x="words length", hue='sentiment', fill=True, palette=[sns.color_palette()[0], 'red']
    ).set(title="Words in review")
    plt.legend(title='Sentiment', labels=['negative', 'positive'])
    plt.show()

    # Split train, test data
    label_encoder = LabelEncoder()
    y_data = label_encoder.fit_transform(df['sentiment'])
    X_data = df['review'].values

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

    # represent document into vector
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(X_train, y_train)

    X_train_encoded = tfidf_vectorizer.transform(X_train)
    X_test_encoded = tfidf_vectorizer.transform(X_test)

    print("----Xtrain encoded: ", X_train_encoded[0])

    # training phase
    dt_cls = DecisionTreeClassifier(criterion='entropy', random_state=42)
    dt_cls.fit(X_train_encoded, y_train)
    y_pred = dt_cls.predict(X_test_encoded)
    print('----DecisionTreeClassifier accuracy score: ', accuracy_score(y_pred, y_test))

    rf_cls = RandomForestClassifier(random_state=42)
    rf_cls.fit(X_train_encoded, y_train)
    y_pred = rf_cls.predict(X_test_encoded)
    print('----RandomForestClassifier accuracy score: ', accuracy_score(y_pred, y_test))

    ada_cls = AdaBoostClassifier(random_state=42)
    ada_cls.fit(X_train_encoded, y_train)
    y_pred = ada_cls.predict(X_test_encoded)
    print('----AdaBoostClassifier accuracy score: ', accuracy_score(y_pred, y_test))

    xgboost_cls = XGBClassifier()
    xgboost_cls.fit(X_train_encoded, y_train)
    y_pred = xgboost_cls.predict(X_test_encoded)
    print('----XGBClassifier accuracy score: ', accuracy_score(y_pred, y_test))

if __name__ == "__main__":
    main()
