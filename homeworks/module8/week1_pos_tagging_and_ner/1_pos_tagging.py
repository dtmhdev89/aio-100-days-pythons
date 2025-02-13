import os
from typing import List
import numpy as np
import torch
import evaluate
from sklearn.model_selection import train_test_split
import nltk
nltk.download('treebank')
from transformers import AutoTokenizer, AutoModelForTokenClassification, \
    TrainingArguments, Trainer, AutoConfig
from torch.utils.data import Dataset
from collections import defaultdict


MAX_LEN = 256

def closure_compute_metrics(accuracy, ignore_label):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        mask = labels != ignore_label
        predictions = np.argmax(predictions, axis=-1)

        return accuracy.compute(
            predictions=predictions[mask],
            references=labels[mask])

    return compute_metrics


class PosTagging_Dataset(Dataset):
    def __init__(
        self,
        sentences: List[List[str]],
        tags: List[List[str]],
        tokenizer,
        label2id,
        max_len=MAX_LEN
    ) -> None:
        super().__init__()

        self.sentences = sentences
        self.tags = tags
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        input_token = self.sentences[idx]
        label_token = self.tags[idx]

        input_token = self.tokenizer.convert_tokens_to_ids(input_token)
        attention_mask = [1] * len(input_token)
        labels = [self.label2id[token] for token in label_token]

        return {
            "input_ids": self.pad_and_truncate(
                input_token,
                pad_id=self.tokenizer.pad_token_id),
            "labels": self.pad_and_truncate(labels, pad_id=self.label2id["0"]),
            "attention_mask": self.pad_and_truncate(attention_mask, pad_id=0)
        }
    
    def pad_and_truncate(self, inputs: List[int], pad_id: int):
        if len(inputs) < self.max_len:
            padded_inputs = inputs + [pad_id] * (self.max_len - len(inputs))
        else:
            padded_inputs = inputs[:self.max_len]
        
        return torch.as_tensor(padded_inputs)


if __name__ == "__main__":
    # Load dataset
    tagged_sentences = nltk.corpus.treebank.tagged_sents()
    print("Number of samples:", len(tagged_sentences))

    sentences, sentence_tags = [], []

    for tagged_sentence in tagged_sentences:
        sentence, tags = zip(*tagged_sentence)
        sentences.append([word.lower() for word in sentence])
        sentence_tags.append([tag for tag in tags])

    # Preprocessing
    train_sentences, test_sentences, \
        train_tags, test_tags = train_test_split(
            sentences,
            sentence_tags,
            test_size=0.3
        )

    valid_sentences, test_sentences, \
        valid_tags, test_tags = train_test_split(
            test_sentences,
            test_tags,
            test_size=0.5
        )

    # Build dataset
    model_name = 'QCRI/bert-base-multilingual-cased-pos-english'
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True
    )

    # Modeling
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    label2id = defaultdict(int, model.config.label2id)
    id2label = {v: k for k, v in label2id.items()}

    # Dataset loader
    train_dataset = PosTagging_Dataset(
        train_sentences,
        train_tags,
        tokenizer=tokenizer,
        label2id=label2id
    )
    val_dataset = PosTagging_Dataset(
        valid_sentences,
        valid_tags,
        tokenizer=tokenizer,
        label2id=label2id
    )
    test_dataset = PosTagging_Dataset(
        test_sentences,
        test_tags,
        tokenizer=tokenizer,
        label2id=label2id
    )

    # Metric
    accuracy = evaluate.load('accuracy')
    ignore_label = len(label2id)

    # Trainer
    training_args = TrainingArguments(
        output_dir="out_dir",
        learning_rate=1e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True
    )

    compute_metrics = closure_compute_metrics(
        accuracy=accuracy,
        ignore_label=ignore_label
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Inference
    # # Tokenization
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_sentence = "We are exploring the topic of deep learning"
    tokens = [tokenizer.convert_tokens_to_ids(test_sentence.split())]
    input = torch.as_tensor(tokens)
    input = input.to(device)

    # # Prediction
    outputs = model(input)
    _, preds = torch.max(outputs.logits, -1)
    preds = preds[0].cpu().numpy()

    # # Decode
    pred_tags = ""
    for pred in preds:
        pred_tags += id2label[pred] + " "

    print(pred_tags)
