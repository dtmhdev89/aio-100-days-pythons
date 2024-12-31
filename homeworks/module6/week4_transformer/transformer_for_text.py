import sys
import os

up_levels = [".."] * 3
PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    *up_levels
))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.text_preprocess_util import preprocess_text, yield_tokens
import time
import torchtext
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset
import re
import string
import torchtext
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import AutoConfig, AutoModelForSequenceClassification, \
    AutoTokenizer, TrainingArguments, Trainer, \
    pipeline
import numpy as np
import evaluate
import matplotlib.pyplot as plt


# Transformer Encoder
# # Token and Positional Embedding
class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_length, device='cpu'):
        super().__init__()
        self.device = device
        self.word_emb = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim
        )
        self.pos_emb = nn.Embedding(
            num_embeddings=max_length,
            embedding_dim=embed_dim
        )

    def forward(self, x):
        N, seq_len = x.size()
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        output1 = self.word_emb(x)
        output2 = self.pos_emb(positions)
        output = output1 + output2
        return output


# # Transformer Encoder Block
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=ff_dim, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=ff_dim, out_features=embed_dim, bias=True)
        )
        self.layernorm_1 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)
        self.layernorm_2 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        attn_output, _ = self.attn(query, key, value)
        attn_output = self.dropout_1(attn_output)
        out_1 = self.layernorm_1(query + attn_output)
        ffn_output = self.ffn(out_1)
        ffn_output = self.dropout_2(ffn_output)
        out_2 = self.layernorm_2(out_1 + ffn_output)
        return out_2


# # Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self,
                 source_vocab_size,
                 embed_dim,
                 max_length,
                 num_layers,
                 num_heads,
                 ff_dim,
                 dropout=0.1,
                 device='cpu') -> None:
        super().__init__()
        self.embedding = TokenAndPositionEmbedding(
            source_vocab_size,
            embed_dim,
            max_length,
            device
        )
        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout
                ) for i in range(num_layers)
            ]
        )

    def forward(self, x):
        output = self.embedding(x)
        for layer in self.layers:
            output = layer(output, output, output)

        return output


# Transformer Decoder
# # Transformer Decoder Block
class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=ff_dim, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=ff_dim, out_features=embed_dim, bias=True)
        )
        self.layernorm_1 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)
        self.layernorm_2 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)
        self.layernorm_3 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.dropout_3 = nn.Dropout(p=dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output, _ = self.attn(x, x, x, attn_mask=tgt_mask)
        attn_output = self.dropout_1(attn_output)
        out_1 = self.layernorm_1(x + attn_output)

        attn_output, _ = self.cross_attn(
            out_1, enc_output, enc_output, attn_mask=src_mask
        )
        attn_output = self.dropout_2(attn_output)
        out_2 = self.layernorm_2(out_1 + attn_output)

        ffn_output = self.ffn(out_2)
        ffn_output = self.dropout_2(ffn_output)
        out_3 = self.layernorm_2(out_2 + ffn_output)
        return out_3


# # Transformer Decoder
class TransformerDecoder(nn.Module):
    def __init__(self,
                 tgt_vocab_size,
                 embed_dim,
                 max_length,
                 num_layers,
                 num_heads,
                 ff_dim,
                 dropout=0.1,
                 device='cpu') -> None:
        super().__init__()
        self.embedding = TokenAndPositionEmbedding(
            vocab_size=tgt_vocab_size,
            embed_dim=embed_dim,
            max_length=max_length,
            device=device
        )
        self.layers = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout
                ) for i in range(num_layers)
            ]
        )

    def forward(self, x, enc_output, src_mask, tgt_mask):
        output = self.embedding(x)
        for layer in self.layers:
            output = layer(output, enc_output, src_mask, tgt_mask)

        return output


# Transformer
class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 embed_dim,
                 max_length,
                 num_layers,
                 num_heads,
                 ff_dim,
                 dropout=0.1,
                 device='cpu') -> None:
        super().__init__()
        self.device = device
        self.encoder = TransformerEncoder(
            source_vocab_size=src_vocab_size,
            embed_dim=embed_dim,
            max_length=max_length,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim
        )
        self.decoder = TransformerDecoder(
            tgt_vocab_size=tgt_vocab_size,
            embed_dim=embed_dim,
            max_length=max_length,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim
        )
        self.fc = nn.Linear(embed_dim, tgt_vocab_size)

    def generate_mask(self, src, tgt):
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]

        src_mask = torch.zeros(
            (src_seq_len, src_seq_len),
            device=self.device
        ).type(torch.bool)

        tgt_mask = (torch.triu(torch.ones(
            (tgt_seq_len, tgt_seq_len),
            device=self.device
        )) == 1).transpose(0, 1)

        tgt_mask = tgt_mask.float().masked_fill(
            tgt_mask == 0, float('-inf')
        ).masked_fill(
            tgt_mask == 1, float(0.0)
        )

        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        enc_output = self.encoder(src)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        output = self.fc(dec_output)

        return output


# Preprocessing
def prepare_dataset(df):
    # create iterator for dataset: (sentence, label)
    for row in df:
        sentence = row['preprocessed_sentence']
        encoded_sentence = vocabulary(tokenizer(sentence))
        label = row['label']
        yield encoded_sentence, label


# Dataloader
# # Using closure way to handle collate_batch with other than batch argument
def create_collate_batch(seq_length):
    def collate_batch(batch):
        # create inputs, offsets, labels for batch
        sentences, labels = list(zip(*batch))
        encoded_sentences = [
            sentence+([0]* (seq_length-len(sentence))) if len(sentence) < seq_length else sentence[:seq_length]
            for sentence in sentences
        ]

        encoded_sentences = torch.tensor(encoded_sentences, dtype=torch.int64)
        labels = torch.tensor(labels)

        return encoded_sentences, labels

    return collate_batch


# Trainer
def train_epoch(
        model,
        optimizer,
        criterion,
        train_dataloader,
        device,
        epoch=0,
        log_interval=50):
    model.train()
    total_acc, total_count = 0, 0
    losses = []
    start_time = time.time()

    for idx, (inputs, labels) in enumerate(train_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        predictions = model(inputs)

        # compute loss
        loss = criterion(predictions, labels)
        losses.append(loss.item())

        # backward
        loss.backward()
        optimizer.step()
        total_acc += (predictions.argmax(1) == labels).sum().item()
        total_count += labels.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches "
                "| accuracy {:8.3f}".format(
                    epoch, idx, len(train_dataloader), total_acc / total_count
                )
            )
            total_acc, total_count = 0, 0
            start_time = time.time()

    epoch_acc = total_acc / total_count
    epoch_loss = sum(losses) / len(losses)

    return epoch_acc, epoch_loss


def evaluate_epoch(model, criterion, valid_dataloader, device):
    model.eval()
    total_acc, total_count = 0, 0
    losses = []

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(valid_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            predictions = model(inputs)

            loss = criterion(predictions, labels)
            losses.append(loss.item())

            total_acc += (predictions.argmax(1) == labels).sum().item()
            total_count += labels.size(0)

    epoch_acc = total_acc / total_count
    epoch_loss = sum(losses) / len(losses)
    return epoch_acc, epoch_loss


def train(
        model,
        model_name,
        save_model,
        optimizer,
        criterion,
        train_dataloader,
        valid_dataloader,
        num_epochs,
        device):
    train_accs, train_losses = [], []
    eval_accs, eval_losses = [], []
    best_loss_eval = 100
    times = []
    for epoch in range(1, num_epochs+1):
        epoch_start_time = time.time()
        # Training
        train_acc, train_loss = train_epoch(
            model, optimizer, criterion,
            train_dataloader, device, epoch)
        train_accs.append(train_acc)
        train_losses.append(train_loss)

        # Evaluation
        eval_acc, eval_loss = evaluate_epoch(
            model, criterion,
            valid_dataloader, device)
        eval_accs.append(eval_acc)
        eval_losses.append(eval_loss)

        # Save best model
        if eval_loss < best_loss_eval:
            torch.save(model.state_dict(), save_model + f'/{model_name}.pt')

        times.append(time.time() - epoch_start_time)
        # Print loss, acc end epoch
        print("-" * 59)
        print({
            "| End of epoch {:3d} | Time: {:5.2f}s | Train Accuracy {:8.3f} | "
            "Train Loss {:8.3f} "
            "| Valid Accuracy {:8.3f} | Valid Loss {:8.3f} ".format(
                epoch, time.time() - epoch_start_time,
                train_acc, train_loss, eval_acc, eval_loss
            )
        })
        print("-" * 59)

    # Load best model
    model.load_state_dict(torch.load(save_model + f'/{model_name}.pt',
                                     weights_only=True))
    model.eval()
    metrics = {
        'train_accuracy': train_accs,
        'train_loss': train_losses,
        'valid_accuracy': eval_accs,
        'valid_loss': eval_losses,
        'time': times
    }
    return model, metrics


def plot_result(num_epochs, train_accs, eval_accs, train_losses, eval_losses):
    epochs = list(range(num_epochs))
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    axs[0].plot(epochs, train_accs, label="Training")
    axs[0].plot(epochs, eval_accs, label="Evaluation")
    axs[1].plot(epochs, train_losses, label="Training")
    axs[1].plot(epochs, eval_losses, label="Evaluation")
    axs[0].set_xlabel("Epochs")
    axs[1].set_xlabel("Epochs")
    axs[0].set_ylabel("Accuracy")
    axs[1].set_ylabel("Loss")
    plt.legend()
    plt.show()


# Modeling
class TransformerEncoderCls(nn.Module):
    def __init__(self,
                 vocab_size,
                 max_length,
                 num_layers,
                 embed_dim,
                 num_heads,
                 ff_dim,
                 dropout=0.1,
                 device='cpu'):
        super().__init__()
        self.encoder = TransformerEncoder(
            vocab_size,
            embed_dim,
            max_length,
            num_layers,
            num_heads,
            ff_dim,
            dropout,
            device
        )
        self.pooling = nn.AvgPool1d(kernel_size=max_length)
        self.fc1 = nn.Linear(in_features=embed_dim, out_features=20)
        self.fc2 = nn.Linear(in_features=20, out_features=2)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.encoder(x)
        output = self.pooling(output.permute(0, 2, 1)).squeeze()
        output = self.dropout(output)
        output = self.fc1(output)
        output = self.dropout(output)
        output = self.fc2(output)

        return output


def create_preprocess_function(tokenizer, max_seq_length):
    def preprocess_function(examples):
        # Tokenize the texts
        result = tokenizer(
            examples["preprocessed_sentence"],
            padding="max_length",
            max_length=max_seq_length,
            truncation=True
        )
        result["label"] = examples['label']

        return result

    return preprocess_function


def create_compute_metrics(metric):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        result = metric.compute(predictions=predictions, references=labels)
        return result

    return compute_metrics

if __name__ == "__main__":
    run_transformer_classification = False
    if run_transformer_classification:
        # Encoder Experience
        batch_size = 32
        src_vocab_size = 1000
        embed_dim = 200
        max_length = 100
        num_layers = 2
        num_heads = 4
        ff_dim = 256

        input = torch.randint(
            high=2,
            size=(batch_size, max_length),
            dtype=torch.int64
        )

        encoder = TransformerEncoder(
            src_vocab_size, embed_dim, max_length, num_layers, num_heads, ff_dim
        )

        encoded = encoder(input)

        print(encoded.shape)
        print(encoded.numel())

        # Transformer Experience
        batch_size = 128
        src_vocab_size = 1000
        tgt_vocab_size = 2000
        embed_dim = 200
        max_length = 100
        num_layers = 2
        num_heads = 4
        ff_dim = 256

        model = Transformer(
            src_vocab_size, tgt_vocab_size,
            embed_dim, max_length, num_layers, num_heads, ff_dim
        )

        src = torch.randint(
            high=2,
            size=(batch_size, max_length),
            dtype=torch.int64
        )

        tgt = torch.randint(
            high=2,
            size=(batch_size, max_length),
            dtype=torch.int64
        )

        prediction = model(src, tgt)

        print(prediction.shape)

        # Text classification using Transformer Encoder
        ds = load_dataset('thainq107/ntc-scv')
        print(ds)
        # print(ds['train'][:1])

        # Preprocessing
        # # word-based tokenizer
        tokenizer = get_tokenizer("basic_english")

        # # build vocabulary
        vocab_size = 10000
        vocabulary = build_vocab_from_iterator(
            yield_tokens(ds['train']['preprocessed_sentence'], tokenizer),
            max_tokens=vocab_size,
            specials=["<pad>", "<unk>"]
        )
        vocabulary.set_default_index(vocabulary["<unk>"])

        train_dataset = prepare_dataset(ds['train'])
        train_dataset = to_map_style_dataset(train_dataset)

        valid_dataset = prepare_dataset(ds['valid'])
        valid_dataset = to_map_style_dataset(valid_dataset)

        test_dataset = prepare_dataset(ds['test'])
        test_dataset = to_map_style_dataset(test_dataset)

        for row in ds['train']:
            print(vocabulary(tokenizer(row['preprocessed_sentence'])))
            break

        # Dataloader
        seq_length = 100
        batch_size = 128
        collate_batch_fn = create_collate_batch(seq_length=seq_length)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_batch_fn,
        )
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_batch_fn
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_batch_fn
        )

        print(next(iter(train_dataloader)))
        print(len(train_dataloader))

        encoded_sentences, labels = next(iter(train_dataloader))
        print(encoded_sentences.shape)
        print(labels.shape)

        # Text classification
        vocab_size = 10000
        max_length = 100
        embed_dim = 200
        num_layers = 2
        num_heads = 4
        ff_dim = 128
        dropout = 0.1

        model = TransformerEncoderCls(
            vocab_size,
            max_length,
            num_layers,
            embed_dim,
            num_heads,
            ff_dim,
            dropout
        )
        print(encoded_sentences.shape)
        predictions = model(encoded_sentences)
        print(predictions.shape)

        # # Training
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = TransformerEncoderCls(
            vocab_size,
            max_length,
            num_layers,
            embed_dim,
            num_heads,
            ff_dim,
            dropout,
            device
        )
        model.to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.00005)

        num_epochs = 50
        save_model = './model'
        os.makedirs(save_model, exist_ok=True)
        model_name = 'model'

        model, metrics = train(
            model,
            model_name,
            save_model,
            optimizer,
            criterion,
            train_dataloader,
            valid_dataloader,
            num_epochs,
            device
        )

        plot_result(
            num_epochs,
            metrics["train_accuracy"],
            metrics["valid_accuracy"],
            metrics["train_loss"],
            metrics["valid_loss"]
        )

        test_acc, test_loss = evaluate_epoch(model, criterion,
                                             test_dataloader, device)
        test_acc, test_loss

    # Text classification using BERT
    run_bert_classification = True

    if run_bert_classification:
        ds = load_dataset('thainq107/ntc-scv')

        model_name = "distilbert-base-uncased"

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True
        )

        print(tokenizer.model_max_length)

        max_seq_length = 100
        max_seq_length = min(max_seq_length, tokenizer.model_max_length)

        preprocess_function = create_preprocess_function(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length)

        processed_dataset = ds.map(
            preprocess_function,
            batched=True,
            desc="Running tokenizer on dataset",
        )

        print(processed_dataset)
        print(processed_dataset['train'][0])

        num_labels = 2
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            finetuning_task="text-classification"
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config
        )
        print(model)
        metric = evaluate.load("accuracy")
        compute_metrics = create_compute_metrics(metric=metric)
        training_args = TrainingArguments(
            output_dir='ntc-scv-distilbert-base-uncased',
            learning_rate=2e-5,
            per_device_train_batch_size=256,
            per_device_eval_batch_size=256,
            num_train_epochs=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=processed_dataset["train"],
            eval_dataset=processed_dataset["valid"],
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
        )

        trainer.train()
        trainer.evaluate(processed_dataset["test"])

        classifier = pipeline(
            "text-classification",
            model="thainq107/ntc-scv-distilbert-base-uncased"
        )

        print(classifier("quán ăn này ngon quá luôn nè"))
