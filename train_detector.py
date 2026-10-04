import argparse
import os
import torch
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import numpy as np

def load_hc3_dataset(split='train', max_samples=None):
    print(f"Loading Hello-SimpleAI/HC3 dataset (all.jsonl)...")
    dataset = load_dataset("json", data_files="hf://datasets/Hello-SimpleAI/HC3/all.jsonl", split="train")
    
    texts = []
    labels = []
    
    for row in dataset:
        for human_ans in row['human_answers']:
            if human_ans.strip():
                texts.append(human_ans)
                labels.append(0) # 0 for Human
        for ai_ans in row['chatgpt_answers']:
            if ai_ans.strip():
                texts.append(ai_ans)
                labels.append(1) # 1 for AI
                
    df = pd.DataFrame({'text': texts, 'label': labels})
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    if max_samples:
        df = df.head(max_samples)
        
    print(f"Loaded {len(df)} samples ({sum(df['label'] == 0)} Human, {sum(df['label'] == 1)} AI)")
    return df

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    
    # Calculate ROC-AUC using probabilities
    try:
        from scipy.special import softmax
        probs = softmax(pred.predictions, axis=1)[:, 1]
        roc_auc = roc_auc_score(labels, probs)
    except Exception:
        roc_auc = 0.0

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
    }

def main():
    parser = argparse.ArgumentParser(description="Train a custom AI text detector")
    parser.add_argument("--model", type=str, default="roberta-base", help="Base model to fine-tune")
    parser.add_argument("--samples", type=int, default=5000, help="Number of samples to use for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--output_dir", type=str, default="./custom_detector", help="Output directory to save model")
    args = parser.parse_args()

    # 1. Load Data
    df = load_hc3_dataset(split='train', max_samples=args.samples)
    
    # Split into train and eval
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)
    
    # 2. Load Tokenizer & Model
    print(f"Loading tokenizer and model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    
    # 3. Tokenize Data
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

    # 4. Set up Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir='./logs',
        logging_steps=10,
        fp16=torch.cuda.is_available(), # Enable mixed precision if CUDA is available
        lr_scheduler_type="cosine",     # Cosine annealing learning rate scheduler
        warmup_ratio=0.1,               # Warmup steps
        label_smoothing_factor=0.1,     # Label smoothing for regularization
    )

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # 6. Train Model
    print("Starting training...")
    trainer.train()

    # 7. Save the best model
    print(f"Training complete. Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
