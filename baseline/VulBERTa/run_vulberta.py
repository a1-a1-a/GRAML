
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

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# --- Data Processing ---

def cleaner(code):
    ## Remove code comments
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
    
    # Calculate metrics
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

def load_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        try:
            # Try loading as a standard JSON array
            data = json.load(f)
        except json.JSONDecodeError:
            # Fallback to JSONL (line-separated JSON)
            f.seek(0)
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    return df

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

    # 1. Load Tokenizer
    # VulBERTa uses a custom BPE tokenizer. For simplicity and robustness, 
    # we use the base RobertaTokenizerFast if custom files are missing, 
    # OR we try to load the local custom tokenizer if files exist.
    
    logger.info("Loading tokenizer...")
    # Check if local tokenizer files exist
    vocab_path = "./tokenizer/drapgh-vocab.json"
    merges_path = "./tokenizer/drapgh-merges.txt"
    
    if os.path.exists(vocab_path) and os.path.exists(merges_path):
        logger.info(f"Using local custom tokenizer from {vocab_path}")
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors
        from tokenizers.models import BPE
        
        # Correctly load BPE vocab and merges
        vocab, merges = BPE.read_file(vocab_path, merges_path)
        tokenizer_obj = Tokenizer(BPE(vocab, merges, unk_token="<unk>"))
        
        # Wrap into Transformers tokenizer
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

    # 2. Load Data
    logger.info("Loading datasets...")
    train_df = load_data(args.train_file)
    valid_df = load_data(args.valid_file)
    test_df = load_data(args.test_file)

    # Helper to process labels
    def process_labels(val):
        if isinstance(val, str):
            if val.lower() == 'yes': return 1
            if val.lower() == 'no': return 0
            try: return int(val)
            except: return 0
        return int(val)

    # Process dataframes
    for df in [train_df, valid_df, test_df]:
        df['func'] = df['input'].apply(cleaner)
        df['target'] = df['output'].apply(process_labels)

    # 3. Tokenize
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

    # 4. Load Model
    logger.info(f"Loading model: {args.model_name_or_path}")
    # Directly load the specified model. Do NOT fallback to roberta-base silently.
    # If this fails, the user should know about it.
    model = RobertaForSequenceClassification.from_pretrained(
        args.model_name_or_path, 
        num_labels=2,
        ignore_mismatched_sizes=True
    )

    # 5. Train
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        eval_strategy='epoch', # Changed from evaluation_strategy to eval_strategy
        save_strategy='epoch',
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='eval_f1',
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
    
    # 6. Test
    if args.do_test:
        logger.info("Running prediction on test set...")
        test_results = trainer.predict(test_dataset)
        
        metrics = test_results.metrics
        # Rename keys for consistency
        results = {
            'model_name': args.model_name_or_path,
            'test_data_file': args.test_file,
            'test_accuracy': metrics['test_eval_acc'],
            'test_precision': metrics['test_eval_precision'],
            'test_recall': metrics['test_eval_recall'],
            'test_f1': metrics['test_eval_f1']
        }

        # Save results
        output_csv_file = os.path.join(args.output_dir, 'test_results.csv')
        df = pd.DataFrame([results])
        
        # Order columns
        ordered_keys = ['model_name', 'test_data_file', 'test_accuracy', 'test_precision', 'test_recall', 'test_f1']
        df = df[ordered_keys]
        
        if os.path.exists(output_csv_file):
            df.to_csv(output_csv_file, mode='a', header=False, index=False)
        else:
            df.to_csv(output_csv_file, mode='w', header=True, index=False)
            
        # Save txt
        with open(os.path.join(args.output_dir, 'test_results.txt'), 'w') as f:
            for key, val in results.items():
                logger.info(f"{key} = {val}")
                f.write(f"{key} = {val}\n")

if __name__ == "__main__":
    main()
