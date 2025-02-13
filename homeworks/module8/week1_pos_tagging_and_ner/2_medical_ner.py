import os
from typing import List, Dict, Tuple
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForTokenClassification
from transformers import pipeline
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import evaluate
import numpy as np


MAX_LEN = 512


class Preprocessing_Maccrobat:
    def __init__(self, dataset_folder, tokenizer):
        self.file_ids = [f.split(".")[0] for f in os.listdir(dataset_folder) if f.endswith('.txt')]

        self.text_files = [f+".txt" for f in self.file_ids]
        self.anno_files = [f+".ann" for f in self.file_ids]

        self.num_samples = len(self.file_ids)

        self.texts: List[str] = []
        for i in range(self.num_samples):
            file_path = os.path.join(dataset_folder, self.text_files[i])
            with open(file_path, 'r') as f:
                self.texts.append(f.read())

        self.tags: List[Dict[str, str]] = []
        for i in range(self.num_samples):
            file_path = os.path.join(dataset_folder, self.anno_files[i])
            with open(file_path, 'r') as f:
                text_bound_ann = [t.split("\t") for t in f.read().split("\n") if t.startswith("T")]
                text_bound_lst = []
                for text_b in text_bound_ann:
                    label = text_b[1].split(" ")
                    try:
                        _ = int(label[1])
                        _ = int(label[2])
                        tag = {
                            "text": text_b[-1],
                            "label": label[0],
                            "start": label[1],
                            "end": label[2]
                        }
                        text_bound_lst.append(tag)
                    except:
                        pass

                self.tags.append(text_bound_lst)
        self.tokenizer = tokenizer

    def process(self) -> Tuple[List[List[str]], List[List[str]]]:
        input_texts = []
        input_labels = []

        for idx in range(self.num_samples):
            full_text = self.texts[idx]
            tags = self.tags[idx]

            label_offset = []
            continuous_label_offset = []
            for tag in tags:
                offset = list(range(int(tag["start"]), int(tag["end"])+1))
                label_offset.append(offset) # 345
                continuous_label_offset.extend(offset) #  345

            all_offset = list(range(len(full_text)))
            zero_offset = [offset for offset in all_offset if offset not in continuous_label_offset]
            zero_offset = Preprocessing_Maccrobat.find_continuous_ranges(zero_offset) # 012 67

            self.tokens = []
            self.labels = []
            self._merge_offset(full_text, tags, zero_offset, label_offset)
            assert len(self.tokens) == len(self.labels), f"Length of tokens and labels are not equal"

            input_texts.append(self.tokens)
            input_labels.append(self.labels)

        return input_texts, input_labels

    def _merge_offset(self, full_text, tags, zero_offset, label_offset):
        # zero: [[0,1,2], [6,7]] label: [[3,4,5]]
        i = j = 0
        while i < len(zero_offset) and j < len(label_offset):
            if zero_offset[i][0] < label_offset[j][0]:
                self._add_zero(full_text, zero_offset, i)
                i += 1
            else:
                self._add_label(full_text, label_offset, j, tags)
                j += 1

        while i < len(zero_offset):
            self._add_zero(full_text, zero_offset, i)
            i += 1

        while j < len(label_offset):
            self._add_label(full_text, label_offset, j, tags)
            j += 1

    def _add_zero(self, full_text, offset, index):
        start, *_ ,end =  offset[index] if len(offset[index]) > 1 else (offset[index][0], offset[index][0]+1)
        text = full_text[start:end]
        text_tokens = self.tokenizer.tokenize(text)

        self.tokens.extend(text_tokens)
        self.labels.extend(
            ["O"]*len(text_tokens)
        )

    def _add_label(self, full_text, offset, index, tags):
        start, *_ ,end =  offset[index] if len(offset[index]) > 1 else (offset[index][0], offset[index][0]+1)
        text = full_text[start:end]
        text_tokens = self.tokenizer.tokenize(text)

        self.tokens.extend(text_tokens)
        self.labels.extend(
            [f"B-{tags[index]['label']}"] + [f"I-{tags[index]['label']}"]*(len(text_tokens)-1)
        )

    @staticmethod
    def build_label2id(tokens: List[List[str]]):
        label2id = {}
        id_counter = 0
        for token in [token for sublist in tokens for token in sublist]:
            if token not in label2id:
                label2id[token] = id_counter
                id_counter += 1
        return label2id

    @staticmethod
    def find_continuous_ranges(data: List[int]): # [0, 1, 2, 6, 7]
        if not data:
            return []
        ranges = []
        start = data[0] # 0
        prev = data[0] # 0
        for number in data[1:]: # [1, 2, 6, 7]
            if number != prev + 1:
                ranges.append(list(range(start, prev + 1)))
                start = number
            prev = number
        ranges.append(list(range(start, prev + 1)))
        return ranges


class NER_Dataset(Dataset):
    def __init__(self, input_texts, input_labels, tokenizer, label2id, max_len=MAX_LEN):
        super().__init__()
        self.tokens = input_texts
        self.labels = input_labels
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        input_token = self.tokens[idx]
        label_token = [self.label2id[label] for label in self.labels[idx]]

        input_token = self.tokenizer.convert_tokens_to_ids(input_token)
        attention_mask = [1] * len(input_token)

        input_ids = self.pad_and_truncate(input_token, pad_id= self.tokenizer.pad_token_id)
        labels = self.pad_and_truncate(label_token, pad_id=0)
        attention_mask = self.pad_and_truncate(attention_mask, pad_id=0)

        return {
            "input_ids": torch.as_tensor(input_ids),
            "labels": torch.as_tensor(labels),
            "attention_mask": torch.as_tensor(attention_mask)
            }

    def pad_and_truncate(self, inputs: List[int], pad_id: int):
        if len(inputs) < self.max_len:
            padded_inputs = inputs + [pad_id] * (self.max_len - len(inputs))
        else:
            padded_inputs = inputs[:self.max_len]
        return padded_inputs

    def label2id(self, labels: List[str]):
        return [self.label2id[label] for label in labels]


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    mask = labels != 0
    predictions = np.argmax(predictions, axis=-1)
    return accuracy.compute(predictions=predictions[mask], references=labels[mask])


def inference(tokenizer, sentence, model, device):
    input = torch.as_tensor([tokenizer.convert_tokens_to_ids(sentence.split())])
    input = input.to(device)
    with torch.no_grad():
        outputs = model(input)
    _, preds = torch.max(outputs.logits, -1)
    preds = preds[0].cpu().numpy()

    return preds


def merge_entity(sentence, preds, model):
    merged_list = []
    prev_value = None
    temp_keys = []

    for key, value in zip(sentence.split(), preds):
        value = model.config.id2label[value].split('-')[-1]
        if value == "O":
            if temp_keys:
                merged_list.append((prev_value, ", ".join(temp_keys)))
                temp_keys = []
            merged_list.append((value, key))
            prev_value = None
        elif value == prev_value:
            temp_keys.append(key)
        else:
            if temp_keys:
                merged_list.append((prev_value, " ".join(temp_keys)))
            temp_keys = [key]
            prev_value = value

    if temp_keys:
        merged_list.append((prev_value, ", ".join(temp_keys)))

    return merged_list


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained("d4data/biomedical-ner-all")

    dataset_folder = os.path.join("MACCROBAT2018")

    Maccrobat_builder = Preprocessing_Maccrobat(dataset_folder, tokenizer)
    input_texts, input_labels = Maccrobat_builder.process()

    label2id = Preprocessing_Maccrobat.build_label2id(input_labels)
    id2label = {v: k for k, v in label2id.items()}

    inputs_train, inputs_val, labels_train, labels_val = train_test_split(
        input_texts,
        input_labels,
        test_size=0.2,
        random_state=42
    )

    train_set = NER_Dataset(inputs_train, labels_train, tokenizer, label2id)
    val_set = NER_Dataset(inputs_val, labels_val, tokenizer, label2id)

    model_name = "d4data/biomedical-ner-all"
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True
    )

    print(model)

    os.environ["WANDB_DISABLED"] = "true"

    accuracy = evaluate.load("accuracy")

    result_output_dir = os.path.join("ner-biomedical-maccrobat2018")
    os.makedirs(result_output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=result_output_dir,
        learning_rate=1e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        optim="adamw_torch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Push to your hugging face
    # trainer.push_to_hub(
    #     commit_message="Training complete",
    #     token=<your_hugging_face_token>
    # )

    sentence = """A 48 year - old female presented with vaginal bleeding and abnormal Pap smears .
    Upon diagnosis of invasive non - keratinizing SCC of the cervix ,
    she underwent a radical hysterectomy with salpingo - oophorectomy
    which demonstrated positive spread to the pelvic lymph nodes and the parametrium .
    Pathological examination revealed that the tumour also extensively involved the lower uterine segment .
    """
    preds = inference(
        tokenizer=tokenizer,
        sentence=sentence,
        model=model,
        device=device
    )
    results = merge_entity(sentence, preds, model)

    print("results: ", results)

    model_checkpoint = "thainq107/ner-biomedical-maccrobat2018"
    token_classifier = pipeline(
        "token-classification",
        model=model_checkpoint,
        aggregation_strategy="simple"
    )
    results = token_classifier(sentence)

    print("from checkpoint model: ", results)

