import pandas as pd
import numpy as np
import torch
import os
import random
import re
import json
import argparse
import logging
from typing import List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from transformers import (
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    RobertaTokenizerFast,
    PreTrainedTokenizerFast
)
from torch.utils.data import Dataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def cleaner(code):
    pat = re.compile(r'(/\*([^*]|(\*+[^*/]))*\*+/)|(//.*)')
    code = re.sub(pat,'',code)
    code = re.sub('\n','',code)
    code = re.sub('\t','',code)
    return(code)

class MyCustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    return {
        'eval_acc': acc,
        'eval_precision': precision,
        'eval_recall': recall,
        'eval_f1': f1,
    }


def find_best_threshold(labels, logits):
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.arange(0.01, 0.91, 0.01):
        preds = (probs >= threshold).astype(int)
        score = f1_score(labels, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold, float(best_f1)

def load_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

    df = pd.DataFrame(data)
    return df


def filter_detection_df(df):
    if df is None or df.empty:
        return df
    task_key = 'Task' if 'Task' in df.columns else ('task' if 'task' in df.columns else None)
    if task_key is not None:
        task = df[task_key].astype(str).str.strip().str.lower()
        df = df[(task == '') | (task == 'detection')]
    if 'output' in df.columns:
        label = df['output'].astype(str).str.strip().str.lower()
        df = df[label.isin(['yes', 'no'])]
    return df.copy()


def main():
    parser = argparse.ArgumentParser(description="Fine-tune VulBERTa")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--valid_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, default="claudios/VulBERTa-MLP-Devign")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run testing.")

    args = parser.parse_args()

    set_seed(args.seed)


    logger.info("Loading tokenizer...")
    vocab_path = "./tokenizer/drapgh-vocab.json"
    merges_path = "./tokenizer/drapgh-merges.txt"

    if os.path.exists(vocab_path) and os.path.exists(merges_path):
        logger.info(f"Using local custom tokenizer from {vocab_path}")
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors
        from tokenizers.models import BPE

        vocab, merges = BPE.read_file(vocab_path, merges_path)
        tokenizer_obj = Tokenizer(BPE(vocab, merges, unk_token="<unk>"))

        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_obj)
        tokenizer.pad_token = "<pad>"
        tokenizer.unk_token = "<unk>"
        tokenizer.mask_token = "<mask>"
        tokenizer.cls_token = "<s>"
        tokenizer.sep_token = "</s>"
        tokenizer.bos_token = "<s>"
        tokenizer.eos_token = "</s>"
    else:
        logger.info("Local tokenizer not found, using 'roberta-base' tokenizer as fallback")
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    logger.info("Loading datasets...")
    train_df = filter_detection_df(load_data(args.train_file))
    valid_df = filter_detection_df(load_data(args.valid_file))
    test_df = filter_detection_df(load_data(args.test_file))
    if train_df.empty or valid_df.empty or test_df.empty:
        raise ValueError("No detection records with Yes/No labels found in the input files")

    def process_labels(val):
        if isinstance(val, str):
            if val.lower() == 'yes': return 1
            if val.lower() == 'no': return 0
            try: return int(val)
            except: return 0
        return int(val)

    for df in [train_df, valid_df, test_df]:
        df['func'] = df['input'].apply(cleaner)
        df['target'] = df['output'].apply(process_labels)

    logger.info("Tokenizing...")
    def tokenize_data(texts):
        return tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=args.block_size,
            return_tensors="pt"
        )

    train_encodings = tokenize_data(train_df['func'].tolist())
    valid_encodings = tokenize_data(valid_df['func'].tolist())
    test_encodings = tokenize_data(test_df['func'].tolist())

    train_dataset = MyCustomDataset(train_encodings, train_df['target'].tolist())
    valid_dataset = MyCustomDataset(valid_encodings, valid_df['target'].tolist())
    test_dataset = MyCustomDataset(test_encodings, test_df['target'].tolist())

    logger.info(f"Loading model: {args.model_name_or_path}")
    model = RobertaForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=2,
        ignore_mismatched_sizes=True
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,
        load_best_model_at_end=False,
        learning_rate=args.learning_rate,
        seed=args.seed,
        fp16=torch.cuda.is_available(),
        logging_dir=os.path.join(args.output_dir, 'logs'),
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )

    if args.do_train:
        logger.info("Starting training...")
        trainer.train()

    if args.do_test:
        logger.info("Selecting threshold on validation set...")
        valid_results = trainer.predict(valid_dataset)
        best_threshold, best_valid_f1 = find_best_threshold(valid_results.label_ids, valid_results.predictions)
        logger.info("Selected threshold %.4f (validation F1 %.4f)", best_threshold, best_valid_f1)

        logger.info("Running prediction on test set...")
        test_results = trainer.predict(test_dataset)
        test_probs = torch.softmax(torch.tensor(test_results.predictions), dim=-1)[:, 1].numpy()
        test_preds = (test_probs >= best_threshold).astype(int)
        test_labels = test_results.label_ids

        results = {
            'model_name': args.model_name_or_path,
            'test_data_file': args.test_file,
            'test_accuracy': accuracy_score(test_labels, test_preds),
            'test_precision': precision_score(test_labels, test_preds, zero_division=0),
            'test_recall': recall_score(test_labels, test_preds, zero_division=0),
            'test_f1': f1_score(test_labels, test_preds, zero_division=0),
            'test_threshold': best_threshold,
        }

        output_csv_file = os.path.join(args.output_dir, 'test_results.csv')
        df = pd.DataFrame([results])

        ordered_keys = ['model_name', 'test_data_file', 'test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_threshold']
        df = df[ordered_keys]

        if os.path.exists(output_csv_file):
            df.to_csv(output_csv_file, mode='a', header=False, index=False)
        else:
            df.to_csv(output_csv_file, mode='w', header=True, index=False)

        with open(os.path.join(args.output_dir, 'test_results.txt'), 'w') as f:
            for key, val in results.items():
                logger.info(f"{key} = {val}")
                f.write(f"{key} = {val}\n")

if __name__ == "__main__":
    main()
