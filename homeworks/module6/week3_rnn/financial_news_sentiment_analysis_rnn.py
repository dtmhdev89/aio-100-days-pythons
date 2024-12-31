import sys
import os

up_levels = [".."] * 3
PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    *up_levels
))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import gdown
from zipfile import ZipFile
import torch
import torch.nn as nn

seed = 1
torch.manual_seed(seed)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
import unidecode

from utils.nltk_util import nltk_download
nltk_download(['stopwords'])

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


def text_normalize(text, english_stop_words, stemmer):
    text = text.lower()
    text = unidecode.unidecode(text)
    text = text.strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(
        [word for word in text.split(' ') if word not in english_stop_words]
    )
    text = ' '.join([stemmer.stem(word) for word in text.split(' ')])

    return text


def transform(text, word_to_idx, max_seq_len):
    tokens = []
    for w in text.split():
        try:
            w_ids = word_to_idx[w]
        except BaseException:
            w_ids = word_to_idx['UNK']
        tokens.append(w_ids)

    if len(tokens) < max_seq_len:
        tokens += [word_to_idx['PAD']] * (max_seq_len - len(tokens))
    elif len(tokens) > max_seq_len:
        tokens = tokens[:max_seq_len]

    return tokens


class FinancialNews(Dataset):
    def __init__(
        self,
        X, y,
        word_to_idx,
        max_seq_len,
        transform=None
    ):
        self.texts = X
        self.labels = y
        self.word_to_idx = word_to_idx
        self.max_seq_len = max_seq_len
        self.transform = transform

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        if self.transform:
            text = self.transform(
                text,
                self.word_to_idx,
                self.max_seq_len
            )
        text = torch.tensor(text)

        return text, label
    

class SentimentClassifier(nn.Module):
    def __init__(
        self, vocab_size, embedding_dim,
        hidden_size, n_layers, n_classes,
        dropout_prob
    ):
        super(SentimentClassifier, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim
        )
        self.rnn = nn.RNN(
            embedding_dim,
            hidden_size,
            n_layers,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(hidden_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, n_classes)

    def forward(self, x):
        x = self.embedding(x)
        x, hn = self.rnn(x)
        x = x[:, -1, :]
        x = self.norm(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


def evaluate(model, dataloader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    losses = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    loss = sum(losses) / len(losses)
    acc = correct / total

    return loss, acc


def fit(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs
):
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        batch_train_losses = []

        model.train()
        for idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_train_losses.append(loss.item())

        train_loss = sum(batch_train_losses) / len(batch_train_losses)
        train_losses.append(train_loss)

        val_loss, val_acc = evaluate(
            model, val_loader,
            criterion, device
        )
        val_losses.append(val_loss)

        print(f'EPOCH {epoch + 1}:\tTrain loss: {train_loss:.4f}\tVal loss: {val_loss:.4f}')

    return train_losses, val_losses


if __name__ == "__main__":
    base_path = os.path.join('./')
    download_path = gdown.download(
        id='1uYXI4O3oWBA6QC8ZJ-r6yaTTfkdAnl_Q',
        output=base_path
    )

    with ZipFile(download_path, 'r') as zip_ref:
        for member in zip_ref.namelist()[1:]:
            if member:
                zip_ref.extract(member, path=base_path)

    dataset_path = os.path.join(base_path, 'dataset', 'all-data.csv')
    headers = ['sentiment', 'content']
    df = pd.read_csv(
        dataset_path,
        names=headers,
        encoding='ISO-8859-1'
    )
    print(df.head())

    classes = {
        class_name: idx for idx, class_name
        in enumerate(df['sentiment'].unique().tolist())
    }
    df['sentiment'] = df['sentiment'].apply(lambda x: classes[x])
    print(df.head())

    english_stop_words = stopwords.words('english')
    stemmer = PorterStemmer()

    df['content'] = df['content'].apply(
        lambda x: text_normalize(
                      x,
                      english_stop_words=english_stop_words,
                      stemmer=stemmer
                  )
    )
    print(df.head())

    vocab = []
    for sentence in df['content'].tolist():
        tokens = sentence.split()
        for token in tokens:
            if token not in vocab:
                vocab.append(token)

    vocab.append('UNK')
    vocab.append('PAD')
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    vocab_size = len(vocab)
    print(vocab[-4:])

    val_size = 0.2
    test_size = 0.125
    is_shuffle = True
    texts = df['content'].tolist()
    labels = df['sentiment'].tolist()

    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels,
        test_size=val_size,
        random_state=seed,
        shuffle=is_shuffle
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train,
        test_size=test_size,
        random_state=seed,
        shuffle=is_shuffle
    )

    max_seq_len = 32

    train_dataset = FinancialNews(
        X_train, y_train,
        word_to_idx=word_to_idx,
        max_seq_len=max_seq_len,
        transform=transform
    )
    val_dataset = FinancialNews(
        X_val, y_val,
        word_to_idx=word_to_idx,
        max_seq_len=max_seq_len,
        transform=transform
    )
    test_dataset = FinancialNews(
        X_test, y_test,
        word_to_idx=word_to_idx,
        max_seq_len=max_seq_len,
        transform=transform
    )

    train_batch_size = 128
    test_batch_size = 8

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=test_batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False
    )

    n_classes = len(list(classes.keys()))
    embedding_dim = 64
    hidden_size = 64
    n_layers = 2
    dropout_prob = 0.2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = SentimentClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        n_layers=n_layers,
        n_classes=n_classes,
        dropout_prob=dropout_prob
    ).to(device)

    model.eval()

    dummy_tensor = torch.randint(10, (1, 10)).to(device)

    with torch.no_grad():
        output = model(dummy_tensor)

    print('Output shape:', output.shape)

    lr = 1e-4
    epochs = 50

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    train_losses, val_losses = fit(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        epochs
    )

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(train_losses)
    ax[0].set_title('Training Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[1].plot(val_losses, color='orange')
    ax[1].set_title('Val Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    plt.show()

    val_loss, val_acc = evaluate(
        model,
        val_loader,
        criterion,
        device
    )
    test_loss, test_acc = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    print('Evaluation on val/test dataset')
    print('Val accuracy: ', val_acc)
    print('Test accuracy: ', test_acc)
