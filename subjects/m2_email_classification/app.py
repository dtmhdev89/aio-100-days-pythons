import os
import string
import nltk
# add custom path to nltk
from nltk.data import path as nltk_path
nltk_download_path = os.path.join(os.getcwd(), 'nltk_data')
nltk_path.append(nltk_download_path)
nltk.download('stopwords', download_dir=nltk_download_path)
nltk.download('punkt', download_dir=nltk_download_path)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

def lowercase(text):
    return text.lower()

def punctuation_removal(text):
    translator = str.maketrans('', '', string.punctuation)

    return text.translate(translator)

def tokenize(text):
    return nltk.word_tokenize(text)

def remove_stopwords(tokens):
    stop_words = nltk.corpus.stopwords.words('english')

    return [token for token in tokens if token not in stop_words]

def stemming(tokens):
    stemmer = nltk.PorterStemmer()

    return [stemmer.stem(token) for token in tokens]

def preprocess_text(text):
    text = lowercase(text)
    text = punctuation_removal(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = stemming(tokens)

    return tokens

def create_dictionary(messages):
    dictionary = []
    
    for tokens in messages:
        for token in tokens:
            if token not in dictionary:
                dictionary.append(token)

    return dictionary

def create_features(tokens, dictionary):
    features = np.zeros(len(dictionary))

    for token in tokens:
        if token in dictionary:
            features[dictionary.index(token)] += 1

    return features

def predict(text, model, dictionary, le):
    processed_text = preprocess_text(text)
    features = create_features(processed_text, dictionary)
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    prediction_cls = le.inverse_transform(prediction)[0]

    return prediction_cls

def main():
    DATASET_PATH = '2cls_spam_text_cls.csv'
    df = pd.read_csv(DATASET_PATH)

    messages = df['Message'].values.tolist()
    labels = df['Category'].values.tolist()

    messages = [preprocess_text(message) for message in messages]
    dictionary = create_dictionary(messages)

    messages_features = [create_features(tokens, dictionary) for tokens in messages]
    X = np.array(messages_features)

    le = LabelEncoder()
    y = le.fit_transform(labels)

    # split train/val/test 7/2/1
    VAL_SIZE = 0.2
    TEST_SIZE = 0.125
    SEED = 0

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VAL_SIZE, shuffle=True, random_state=SEED)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=TEST_SIZE, shuffle=True, random_state=SEED)

    model = GaussianNB()
    print('Starting training...')
    model = model.fit(X_train, y_train)
    print('Training completed')

    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    val_accuracy = accuracy_score(y_val, y_val_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Val accuracy: {val_accuracy}")
    print(f"Test accuracy: {test_accuracy}")

    test_input = 'I am actually thinking a way of doing something useful'
    prediction_cls = predict(test_input, model, dictionary, le)
    print(f"Prediction: {prediction_cls}")

if __name__ == "__main__":
    main()
