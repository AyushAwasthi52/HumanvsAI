import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import uuid
import optuna
import datasets.arrow_dataset
import datasets.fingerprint

# Compatibility patch for Python 3.14 pickle changes with HuggingFace datasets fingerprinting
datasets.arrow_dataset.generate_fingerprint = lambda *a, **k: uuid.uuid4().hex
datasets.fingerprint.generate_fingerprint = lambda *a, **k: uuid.uuid4().hex
datasets.fingerprint.update_fingerprint = lambda *a, **k: uuid.uuid4().hex

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# ==============================================================================
# ADVANCED LOSS FUNCTION: FOCAL LOSS
# Addresses hard-to-classify examples by down-weighting easily classified ones.
# ==============================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.logits
        # Use custom Focal Loss instead of standard Cross Entropy
        loss_fct = FocalLoss(alpha=1.0, gamma=2.0)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# ==============================================================================
# METRICS COMPUTATION
# ==============================================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    preds = logits.argmax(axis=-1)
    
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "roc_auc": roc_auc_score(labels, probs)
    }

def load_and_tokenize_data(data_dir, tokenizer_name="roberta-base", max_train_samples=None, max_val_samples=None):
    print("Loading Parquet datasets...")
    train_df = pd.read_parquet(os.path.join(data_dir, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(data_dir, "val.parquet"))
    
    if max_train_samples is not None and max_train_samples < len(train_df):
        print(f"Subsampling train set to {max_train_samples} samples (stratified)...")
        sub_0 = train_df[train_df['label'] == 0].sample(n=min(len(train_df[train_df['label'] == 0]), max_train_samples // 2), random_state=42)
        sub_1 = train_df[train_df['label'] == 1].sample(n=min(len(train_df[train_df['label'] == 1]), max_train_samples - len(sub_0)), random_state=42)
        train_df = pd.concat([sub_0, sub_1]).sample(frac=1, random_state=42).reset_index(drop=True)
        
    if max_val_samples is not None and max_val_samples < len(val_df):
        print(f"Subsampling val set to {max_val_samples} samples (stratified)...")
        sub_0 = val_df[val_df['label'] == 0].sample(n=min(len(val_df[val_df['label'] == 0]), max_val_samples // 2), random_state=42)
        sub_1 = val_df[val_df['label'] == 1].sample(n=min(len(val_df[val_df['label'] == 1]), max_val_samples - len(sub_0)), random_state=42)
        val_df = pd.concat([sub_0, sub_1]).sample(frac=1, random_state=42).reset_index(drop=True)
        
    print(f"Active dataset size -> Train: {len(train_df)} (Human: {sum(train_df['label']==0)}, AI: {sum(train_df['label']==1)}) | Val: {len(val_df)}")
    
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
        
    print("Tokenizing datasets with dynamic padding...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_val = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    return tokenized_train, tokenized_val, tokenizer

# ==============================================================================
# OPTUNA HYPERPARAMETER TUNING OBJECTIVE
# ==============================================================================
def objective(trial, tokenized_train, tokenized_val, output_dir, epochs=3, batch_size=16, tokenizer=None):
    # Advanced Hyperparameter Search Space
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 3e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.15)
    warmup_steps = trial.suggest_int("warmup_steps", 5, 30)
    lora_r = trial.suggest_categorical("lora_r", [8, 16, 32])
    lora_alpha = trial.suggest_categorical("lora_alpha", [16, 32, 64])
    lora_dropout = trial.suggest_float("lora_dropout", 0.05, 0.2)
    
    print(f"\n[{'='*40}]")
    print(f"STARTING TRIAL {trial.number}")
    print(f"LR: {learning_rate:.6f} | WD: {weight_decay:.3f} | Rank: {lora_r} | Alpha: {lora_alpha} | Warmup Steps: {warmup_steps}")
    
    # 1. Base Architecture
    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base", 
        num_labels=2,
        problem_type="single_label_classification"
    )
    
    # 2. Parameter-Efficient Fine-Tuning (PEFT)
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, 
        r=lora_r, 
        lora_alpha=lora_alpha, 
        lora_dropout=lora_dropout,
        target_modules=["query", "value", "key", "output.dense"] # Deep LoRA integration
    )
    model = get_peft_model(model, peft_config)
    
    # 3. Robust Training Configurations
    trial_dir = os.path.join(output_dir, f"trial_{trial.number}")
    training_args = TrainingArguments(
        output_dir=trial_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2, # Simulates larger effective batch size
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        fp16=True, 
        max_grad_norm=1.0, # Gradient clipping for stability
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        report_to=["tensorboard"], # Professional experiment tracking
        logging_steps=10,
        disable_tqdm=False
    )
    
    # 4. Custom Trainer Execution
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    trainer.train()
    eval_results = trainer.evaluate()
    
    if tokenizer is not None:
        tokenizer.save_pretrained(trial_dir)
        
    return eval_results["eval_roc_auc"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./optuna_models")
    parser.add_argument("--n_trials", type=int, default=10, help="Number of tuning trials")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Max train samples (for quick run)")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Max val samples (for quick run)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs per trial")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    tokenized_train, tokenized_val, tokenizer = load_and_tokenize_data(
        args.data_dir,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples
    )
    
    # Pruner aggressively stops poorly performing trials to save time
    pruner = optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=1)
    study = optuna.create_study(direction="maximize", study_name="RoBERTa_LoRA_Tuning", pruner=pruner)
    
    try:
        study.optimize(
            lambda trial: objective(
                trial, 
                tokenized_train, 
                tokenized_val, 
                args.output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                tokenizer=tokenizer
            ), 
            n_trials=args.n_trials
        )
    except KeyboardInterrupt:
        print("\nOptimization interrupted. Saving best results so far...")
    
    print("\n" + "="*50)
    print("OPTUNA OPTIMIZATION COMPLETE")
    print("Best Trial:")
    print(f"  Trial Number: {study.best_trial.number}")
    print(f"  ROC-AUC: {study.best_trial.value:.5f}")
    print("  Optimal Hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    print(f"Best model weights located in: {os.path.join(args.output_dir, f'trial_{study.best_trial.number}')}")
    print("="*50)

if __name__ == "__main__":
    main()
